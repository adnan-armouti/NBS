// HuPR fused kernels: gain-weighted static scatter (K1) and slow-time
// velocity splat (K2). Semantics pinned by the PyTorch references in
// pipeline/hupr_chirp_motion.py (assemble_static_profile and
// assemble_surface_slow_time_profiles); every kernel is gated against
// them by pipeline tests before it earns training time.
//
// K2 is the training hot path. The torch reference materializes
// (loops, spread, paths) complex tensors (~200 MB per frame per radar at
// person scale) before one giant scatter; the kernel computes each
// contribution in registers and scatters it immediately — nothing is
// materialized. Backward is analytic: per-path gathers, no atomics.

#include <cstdlib>
#include <cuda_runtime.h>
#include <torch/types.h>

#include "utils.cuh"   // ours: see mm4DPS/cuda/utils.cuh
#include "doppler_closed_form.cuh"   // the Hann Doppler PSF in closed form

namespace mm25v5 {

// ---------------------------------------------------------------------
// K1: static profile = scatter of contrib[s,p] * gain[point[p]]
// ---------------------------------------------------------------------

__global__ void static_gain_splat_forward_kernel(
    const float* __restrict__ contrib_real,   // (spread, paths)
    const float* __restrict__ contrib_imag,   // (spread, paths)
    const int* __restrict__ flat_indices, // (spread, paths) into base
    const int64_t* __restrict__ point_indices,// (paths,)
    const float* __restrict__ gains,          // (n_points,)
    float* __restrict__ out_real,             // (base,)
    float* __restrict__ out_imag,             // (base,)
    int64_t spread, int64_t paths)
{
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= spread * paths) return;
    const int64_t p = idx % paths;
    const float g = gains[point_indices[p]];
    const int64_t flat = flat_indices[idx];
    atomicAdd(&out_real[flat], contrib_real[idx] * g);
    atomicAdd(&out_imag[flat], contrib_imag[idx] * g);
}

__global__ void static_gain_splat_backward_kernel(
    const float* __restrict__ grad_out_real,  // (base,)
    const float* __restrict__ grad_out_imag,  // (base,)
    const float* __restrict__ contrib_real,   // (spread, paths)
    const float* __restrict__ contrib_imag,   // (spread, paths)
    const int* __restrict__ flat_indices, // (spread, paths)
    const int64_t* __restrict__ point_indices,// (paths,)
    const float* __restrict__ gains,          // (n_points,)
    float* __restrict__ grad_gains,           // (n_points,) atomic
    float* __restrict__ grad_contrib_real,    // (spread, paths)
    float* __restrict__ grad_contrib_imag,    // (spread, paths)
    int64_t spread, int64_t paths)
{
    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= paths) return;
    const int64_t point = point_indices[p];
    const float g = gains[point];
    // R094: `grad_gains == nullptr` means nobody asked for the gains
    // gradient, the same convention `static_gain_map_backward_kernel` already
    // uses for `grad_contrib`. It is worth a branch: the gains gradient is the
    // ONLY reason this kernel reads `contrib` at all (28 MB) and the only
    // reason it ends in an atomic, and the trainable background never wants it
    // -- there the gains are the frozen areas, not a parameter. Measured in
    // the trainer, this path costs 84.6 us with the gains gradient and the
    // contention is over the real, clustered point distribution.
    // `grad_contrib` is written by exactly the same statements either way, so
    // switching the flag cannot change it by a bit.
    const bool want_gains = (grad_gains != nullptr);
    float g_accum = 0.0f;
    for (int64_t s = 0; s < spread; ++s) {
        const int64_t idx = s * paths + p;
        const int64_t flat = flat_indices[idx];
        const float go_re = grad_out_real[flat];
        const float go_im = grad_out_imag[flat];
        // out = g * c: d/dg = Re(conj(c) grad) pairs; d/dc = g * grad.
        if (want_gains) {
            g_accum += go_re * contrib_real[idx] + go_im * contrib_imag[idx];
        }
        grad_contrib_real[idx] = g * go_re;
        grad_contrib_imag[idx] = g * go_im;
    }
    if (want_gains) atomicAdd(&grad_gains[point], g_accum);
}

// ---------------------------------------------------------------------
// E9b (Phase 5 W2): K1 as a pair of fixed sparse maps.
//
// `contrib`, `flat_indices` and `point_indices` come from the *static scene*:
// they are identical on every step of the whole run and only `gains` -- the
// trained parameter -- changes. So K1 is a fixed linear map `out = S . gains`
// and `S` is built, sorted and prefix-summed once, host-side, and never
// touched again (`hupr_cuda_render.py:_static_gain_map`).
//
// Forward: one block per output cell, a segmented reduction over the ~1,150
// contributions that land in it. Backward: one warp per *point*, a reduction
// over that point's paths. Neither uses an atomicAdd, both sum in an order
// fixed by a map that is built once, so both are deterministic -- which is
// what made `background_gains` the noisiest gradient in every gate table since
// Phase 1.
// ---------------------------------------------------------------------

__global__ void static_gain_map_forward_kernel(
    const int* __restrict__ cell_offsets,     // (base + 1)
    const float2* __restrict__ slot_contrib,  // (nnz,) permuted into cell order
    const int* __restrict__ slot_point,       // (nnz,)
    const float* __restrict__ gains,          // (n_points,)
    float* __restrict__ out_real,             // (base,)
    float* __restrict__ out_imag)
{
    extern __shared__ float2 reduce[];
    const int cell = blockIdx.x;
    const int begin = cell_offsets[cell];
    const int end = cell_offsets[cell + 1];
    float2 accum = make_float2(0.0f, 0.0f);
    for (int slot = begin + threadIdx.x; slot < end; slot += blockDim.x) {
        const float2 c = slot_contrib[slot];
        const float g = gains[slot_point[slot]];
        accum.x += c.x * g;
        accum.y += c.y * g;
    }
    reduce[threadIdx.x] = accum;
    __syncthreads();
    for (int half = blockDim.x >> 1; half > 0; half >>= 1) {
        if (threadIdx.x < half) {
            reduce[threadIdx.x].x += reduce[threadIdx.x + half].x;
            reduce[threadIdx.x].y += reduce[threadIdx.x + half].y;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_real[cell] = reduce[0].x;
        out_imag[cell] = reduce[0].y;
    }
}

// One warp per point. `point_paths` lists each point's paths contiguously, so
// the sum is over a fixed set in a fixed order and lands in a plain store.
__global__ void static_gain_map_backward_kernel(
    const float* __restrict__ grad_out_real,  // (base,)
    const float* __restrict__ grad_out_imag,
    const float* __restrict__ contrib_real,   // (spread, paths)
    const float* __restrict__ contrib_imag,
    const int* __restrict__ flat_indices, // (spread, paths)
    const int* __restrict__ point_offsets,    // (n_points + 1)
    const int* __restrict__ point_paths,      // (paths,)
    const float* __restrict__ gains,          // (n_points,)
    float* __restrict__ grad_gains,           // (n_points,)
    float* __restrict__ grad_contrib_real,    // (spread, paths) or null
    float* __restrict__ grad_contrib_imag,
    int64_t spread, int64_t paths, int64_t n_points)
{
    const int64_t warp = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= n_points) return;
    const int lane = threadIdx.x & 31;
    const int begin = point_offsets[warp];
    const int end = point_offsets[warp + 1];
    const float g = gains[warp];
    float accum = 0.0f;
    for (int entry = begin + lane; entry < end; entry += 32) {
        const int64_t p = point_paths[entry];
        for (int64_t s = 0; s < spread; ++s) {
            const int64_t idx = s * paths + p;
            const int64_t flat = flat_indices[idx];
            const float go_re = grad_out_real[flat];
            const float go_im = grad_out_imag[flat];
            accum += go_re * contrib_real[idx] + go_im * contrib_imag[idx];
            if (grad_contrib_real != nullptr) {
                grad_contrib_real[idx] = g * go_re;
                grad_contrib_imag[idx] = g * go_im;
            }
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        accum += __shfl_down_sync(0xffffffff, accum, offset);
    if (lane == 0) grad_gains[warp] = accum;
}

// ---------------------------------------------------------------------
// K1 for a batch of frames, from the BSDF carrier (docs/TRAINING_SPEED_PLAN_2.md
// 3.1/3.2): the scene's contributions are never materialised.
//
// A contribution is carrier[path] x PSF[s](n_frac[path]) -- the BSDF
// amplitude and carrier phasor (the only factor that moves between steps)
// times the range PSF at the path's fractional bin (a table lerp, frozen
// geometry). The step used to build all (spread, paths) of them, permute
// them into cell order, then run one segmented sum per frame and one
// backward per frame, and finally reduce the per-frame contribution
// gradients over the spread: ~12 ms of a 41 ms step for 2.76 M scene paths.
// Here the forward synthesises each slot's contribution inside the segmented
// sum (the lerp reproduces `_psf_lerp_eager` expression for expression) and
// applies every frame's gain to it at once; the backward is one thread per
// path, walking its `spread` cells in the path-major layout (every read
// coalesced, the gradient reads served from L2) and writing the carrier
// gradient directly. Both sums run in a fixed order, so both reproduce
// themselves bitwise.
// ---------------------------------------------------------------------

constexpr int SCENE_MAX_FRAMES = 8;
constexpr int SCENE_LANES = 8;            // threads per cell in the forward

// `table4` is the PSF table packed for one 16-byte load per lerp:
// (spread, n_grid) float4 of (re[lo], im[lo], re[lo + 1], im[lo + 1]), the
// last column repeating its own entry (`forward_render._psf_table4`).
__device__ __forceinline__ void psf_lerp(
    const float4* __restrict__ table4, int s, float n_frac, int n_grid,
    float* pr, float* pi)
{
    // `_psf_lerp_eager`: idx_f = n_frac * n_grid; lo = clamp(long(idx_f));
    // hi = clamp(lo + 1); frac = idx_f - float(lo); lerp.
    const float idx_f = n_frac * (float)n_grid;
    int lo = (int)idx_f;
    lo = lo < 0 ? 0 : (lo > n_grid - 1 ? n_grid - 1 : lo);
    const float frac = idx_f - (float)lo;
    const float4 t = table4[(int64_t)s * n_grid + lo];
    *pr = t.x * (1.0f - frac) + t.z * frac;
    *pi = t.y * (1.0f - frac) + t.w * frac;
}

// The scene's paths ordered by (channel, range bin): a cell's paths are then
// one or two contiguous runs of that order (two where the PSF wraps at the
// range axis), and the cells of one channel walk overlapping runs, so the
// walk streams through L1 instead of the random 16-byte reads a cell-order
// permutation costs. `entry` is the sorted static record of a path: its path
// index, point, integer range bin and fractional bin; the carrier and the
// point gains arrive gathered into the same order (reading the gains by
// point from L2 instead was measured 1.3 ms slower than the gather).
// `runs` holds each cell's two [begin, end) runs.
__global__ void static_scene_frames_forward_kernel(
    const int4* __restrict__ runs,            // (base,) begin1, end1, begin2, end2
    const int4* __restrict__ entry,           // (paths,) path, point, n_floor, n_frac bits
    const float2* __restrict__ carrier,       // (paths,) in entry order
    const float4* __restrict__ table4,        // (spread, n_grid) packed PSF table
    const float* __restrict__ gains_s,        // (paths, frames) in entry order
    float2* __restrict__ out,                 // (frames, base)
    int64_t base, int K, int half, int n_grid, int frames)
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t cell = tid / SCENE_LANES;
    const int lane = (int)(tid % SCENE_LANES);
    if (cell >= base) return;
    const int bin = (int)(cell % K);
    const int4 r = runs[cell];
    float2 accum[SCENE_MAX_FRAMES];
    #pragma unroll
    for (int f = 0; f < SCENE_MAX_FRAMES; ++f) accum[f] = make_float2(0.0f, 0.0f);
    for (int run = 0; run < 2; ++run) {
        const int begin = run == 0 ? r.x : r.z;
        const int end = run == 0 ? r.y : r.w;
        for (int k = begin + lane; k < end; k += SCENE_LANES) {
            const int4 e = entry[k];
            int dn = bin - (e.z & 0xffff);          // the tap this cell is for this path
            if (dn > half) dn -= K;
            if (dn < -half) dn += K;
            const float2 c = carrier[k];
            float pr, pi;
            psf_lerp(table4, dn + half, __int_as_float(e.w), n_grid, &pr, &pi);
            const float re = c.x * pr - c.y * pi;
            const float im = c.x * pi + c.y * pr;
            const float* g = gains_s + (int64_t)k * frames;
            #pragma unroll
            for (int f = 0; f < SCENE_MAX_FRAMES; ++f) {
                if (f < frames) {
                    accum[f].x += re * g[f];
                    accum[f].y += im * g[f];
                }
            }
        }
    }
    // The cell's lanes reduce in a fixed butterfly: every lane ends with the
    // same sum in the same order, run to run.
    #pragma unroll
    for (int f = 0; f < SCENE_MAX_FRAMES; ++f) {
        if (f < frames) {
            for (int offset = SCENE_LANES / 2; offset > 0; offset >>= 1) {
                accum[f].x += __shfl_xor_sync(0xffffffff, accum[f].x, offset);
                accum[f].y += __shfl_xor_sync(0xffffffff, accum[f].y, offset);
            }
        }
    }
    if (lane == 0)
        for (int f = 0; f < frames; ++f) out[(int64_t)f * base + cell] = accum[f];
}

// One thread per sorted entry: grad_carrier[path] = sum_s conj(PSF_s) . sum_f g_f[point] grad_out_f[cell_s].
// Neighbouring threads hold neighbouring range bins of one channel, so
// their cells share cache lines; the incoming gradient is read transposed,
// (base, frames), so a cell's frames are one sector.
__global__ void static_scene_frames_backward_kernel(
    const float2* __restrict__ grad_out_t,    // (base, frames)
    const int4* __restrict__ entry,           // (paths,) path, point, n_floor, n_frac bits
    const float4* __restrict__ table4,        // (spread, n_grid) packed PSF table
    const float* __restrict__ gains_t,        // (n_points, frames)
    float2* __restrict__ grad_carrier,        // (paths,)
    int64_t paths, int K, int half, int n_grid, int frames)
{
    const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= paths) return;
    const int4 e = entry[k];
    float g[SCENE_MAX_FRAMES];
    #pragma unroll
    for (int f = 0; f < SCENE_MAX_FRAMES; ++f)
        g[f] = (f < frames) ? gains_t[(int64_t)e.y * frames + f] : 0.0f;
    const float nf = __int_as_float(e.w);
    // the cell of tap dn is channel * K + (n_floor + dn) mod K; the entry's
    // third field packs (channel << 16) | n_floor
    const int channel = e.z >> 16;
    const int n_floor = e.z & 0xffff;
    float2 gc = make_float2(0.0f, 0.0f);
    for (int dn = -half; dn <= half; ++dn) {
        int b = n_floor + dn; b %= K; if (b < 0) b += K;
        const float2* go = grad_out_t + ((int64_t)channel * K + b) * frames;
        float2 gs = make_float2(0.0f, 0.0f);
        #pragma unroll
        for (int f = 0; f < SCENE_MAX_FRAMES; ++f) {
            if (f < frames) {
                const float2 v = go[f];
                gs.x += g[f] * v.x;
                gs.y += g[f] * v.y;
            }
        }
        float pr, pi;
        psf_lerp(table4, dn + half, nf, n_grid, &pr, &pi);
        gc.x += gs.x * pr + gs.y * pi;
        gc.y += -gs.x * pi + gs.y * pr;
    }
    grad_carrier[e.x] = gc;
}

// ---------------------------------------------------------------------
// K2: slow-time velocity splat.
// out[l, flat[s,p]] += contrib[s,p] * exp(j * phase_const * rate[p] * t[l, tx[p]])
// ---------------------------------------------------------------------

__global__ void slow_time_splat_forward_kernel(
    const float* __restrict__ contrib_real,   // (spread, paths)
    const float* __restrict__ contrib_imag,   // (spread, paths)
    const int* __restrict__ flat_indices, // (spread, paths) into base
    const int64_t* __restrict__ tx_indices,   // (paths,)
    const float* __restrict__ path_rates,     // (paths,) m/s
    const float* __restrict__ times,          // (loops, n_tx) seconds
    const float phase_const,                  // SIGN * 2*pi / lambda
    float* __restrict__ out_real,             // (loops * base,)
    float* __restrict__ out_imag,
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base)
{
    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= paths) return;
    const float rate = path_rates[p];
    const int64_t tx = tx_indices[p];
    for (int64_t l = 0; l < loops; ++l) {
        const float phase = phase_const * rate * times[l * n_tx + tx];
        float sin_p, cos_p;
        sincosf_reduced(phase, &sin_p, &cos_p);
        const int64_t loop_offset = l * base;
        for (int64_t s = 0; s < spread; ++s) {
            const int64_t idx = s * paths + p;
            const float c_re = contrib_real[idx];
            const float c_im = contrib_imag[idx];
            const int64_t flat = loop_offset + flat_indices[idx];
            atomicAdd(&out_real[flat], c_re * cos_p - c_im * sin_p);
            atomicAdd(&out_imag[flat], c_re * sin_p + c_im * cos_p);
        }
    }
}

__global__ void slow_time_splat_backward_kernel(
    const float* __restrict__ grad_out_real,  // (loops * base,)
    const float* __restrict__ grad_out_imag,
    const float* __restrict__ contrib_real,   // (spread, paths)
    const float* __restrict__ contrib_imag,
    const int* __restrict__ flat_indices, // (spread, paths)
    const int64_t* __restrict__ tx_indices,   // (paths,)
    const float* __restrict__ path_rates,     // (paths,)
    const float* __restrict__ times,          // (loops, n_tx)
    const float phase_const,
    float* __restrict__ grad_contrib_real,    // (spread, paths)
    float* __restrict__ grad_contrib_imag,
    float* __restrict__ grad_rates,           // (paths,)
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base)
{
    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= paths) return;
    const float rate = path_rates[p];
    const int64_t tx = tx_indices[p];
    float rate_accum = 0.0f;
    // grad_contrib rows accumulate over loops; keep them in registers per
    // spread step by looping spread outer / loops inner.
    for (int64_t s = 0; s < spread; ++s) {
        const int64_t idx = s * paths + p;
        const float c_re = contrib_real[idx];
        const float c_im = contrib_imag[idx];
        const int64_t flat = flat_indices[idx];
        float gc_re = 0.0f, gc_im = 0.0f;
        for (int64_t l = 0; l < loops; ++l) {
            const float t = times[l * n_tx + tx];
            const float phase = phase_const * rate * t;
            float sin_p, cos_p;
            sincosf_reduced(phase, &sin_p, &cos_p);
            const float go_re = grad_out_real[l * base + flat];
            const float go_im = grad_out_imag[l * base + flat];
            // out = c * e^{j phi}. With W-linear real pairs:
            //   d/dc_re = go_re*cos + go_im*sin
            //   d/dc_im = -go_re*sin + go_im*cos
            gc_re += go_re * cos_p + go_im * sin_p;
            gc_im += -go_re * sin_p + go_im * cos_p;
            // d out / d phi = j * out => grad_phi =
            //   go_re * (-c_re sin - c_im cos) + go_im * (c_re cos - c_im sin)
            const float d_re = -c_re * sin_p - c_im * cos_p;
            const float d_im = c_re * cos_p - c_im * sin_p;
            rate_accum += (go_re * d_re + go_im * d_im) * phase_const * t;
        }
        grad_contrib_real[idx] = gc_re;
        grad_contrib_imag[idx] = gc_im;
    }
    grad_rates[p] = rate_accum;
}

// ---------------------------------------------------------------------
// K2b (E1): splat straight into the Doppler bins the consumer asked for.
//
// The loop -> Doppler map is linear, so it folds into the splat and the
// kernel writes only the planes that are wanted:
//
//   A[d,p] = sum_l  W[d,l] * e^{j phi(p,l)}
//   out[d, flat[s,p]] += contrib[s,p] * A[d,p]
//
// `W` is a (planes, loops) complex matrix supplied by the caller. Row 0 is
// the one-hot e_0, so plane 0 is loop 0 and that is what the RA term reads.
// Each path's *contribution* to it is bitwise what the 64-plane kernel
// computed (1*x and x+0 are exact); the accumulated plane still differs at
// the atomicAdd-ordering floor, as it does between any two runs. Rows 1..
// are the fftshifted DFT rows for the requested Doppler bins. Nothing is
// truncated: every requested bin gets the full 64-term response from every
// path.
//
// W is built host-side with exact integer angle reduction. Building it in
// float32 instead costs 1.98e-06 on the micro-Doppler marginal -- 10x the
// nondeterminism floor -- because its angles reach ~346 rad, where fp32
// cosf/sinf have already lost 2e-5. That, not the 64-term sum, was the
// whole accuracy question here: with an accurate W a plain fp32
// accumulator lands at 1.40e-07, i.e. at the dense chain's own distance
// from a float64 gold (5.70e-08).
//
// A[.,p] lives in dynamic shared memory: `planes` is a runtime value, and
// a local array indexed dynamically would spill to local memory.
// ---------------------------------------------------------------------

__global__ void doppler_plane_splat_forward_kernel(
    const float* __restrict__ contrib_real,   // (spread, paths)
    const float* __restrict__ contrib_imag,   // (spread, paths)
    const int* __restrict__ flat_indices, // (spread, paths) into base
    const int64_t* __restrict__ tx_indices,   // (paths,)
    const float* __restrict__ path_rates,     // (paths,) m/s
    const float* __restrict__ times,          // (loops, n_tx) seconds
    const float* __restrict__ weight_real,    // (planes, loops)
    const float* __restrict__ weight_imag,    // (planes, loops)
    const float phase_const,                  // SIGN * 2*pi / lambda
    float* __restrict__ out_real,             // (planes * base,)
    float* __restrict__ out_imag,
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base, int64_t planes)
{
    extern __shared__ float2 shared_accum[];
    float2* A = shared_accum + threadIdx.x;    // stride blockDim.x per plane
    const int64_t stride = blockDim.x;

    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= paths) return;
    const float rate = path_rates[p];
    const int64_t tx = tx_indices[p];

    for (int64_t d = 0; d < planes; ++d) A[d * stride] = make_float2(0.0f, 0.0f);
    for (int64_t l = 0; l < loops; ++l) {
        const float phase = phase_const * rate * times[l * n_tx + tx];
        float sin_p, cos_p;
        sincosf_reduced(phase, &sin_p, &cos_p);
        for (int64_t d = 0; d < planes; ++d) {
            const float w_re = weight_real[d * loops + l];
            const float w_im = weight_imag[d * loops + l];
            float2 a = A[d * stride];
            a.x += w_re * cos_p - w_im * sin_p;
            a.y += w_re * sin_p + w_im * cos_p;
            A[d * stride] = a;
        }
    }

    // contrib and flat_indices are read once per spread step, not once per
    // (plane, spread) -- 15 reads instead of 210.
    for (int64_t s = 0; s < spread; ++s) {
        const int64_t idx = s * paths + p;
        const float c_re = contrib_real[idx];
        const float c_im = contrib_imag[idx];
        const int64_t flat = flat_indices[idx];
        for (int64_t d = 0; d < planes; ++d) {
            const float2 a = A[d * stride];
            const int64_t out_index = d * base + flat;
            atomicAdd(&out_real[out_index], c_re * a.x - c_im * a.y);
            atomicAdd(&out_imag[out_index], c_re * a.y + c_im * a.x);
        }
    }
}

__global__ void doppler_plane_splat_backward_kernel(
    const float* __restrict__ grad_out_real,  // (planes * base,)
    const float* __restrict__ grad_out_imag,
    const float* __restrict__ contrib_real,   // (spread, paths)
    const float* __restrict__ contrib_imag,
    const int* __restrict__ flat_indices, // (spread, paths)
    const int64_t* __restrict__ tx_indices,   // (paths,)
    const float* __restrict__ path_rates,     // (paths,)
    const float* __restrict__ times,          // (loops, n_tx)
    const float* __restrict__ weight_real,    // (planes, loops)
    const float* __restrict__ weight_imag,
    const float phase_const,
    float* __restrict__ grad_contrib_real,    // (spread, paths)
    float* __restrict__ grad_contrib_imag,
    float* __restrict__ grad_rates,           // (paths,)
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base, int64_t planes)
{
    // Two shared accumulators per thread: A[d] (as in the forward) and
    // T[d] = sum_s conj(grad_out[d, flat[s,p]]) * contrib[s,p].
    extern __shared__ float2 shared_accum[];
    const int64_t stride = blockDim.x;
    float2* A = shared_accum + threadIdx.x;
    float2* T = shared_accum + planes * stride + threadIdx.x;

    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= paths) return;
    const float rate = path_rates[p];
    const int64_t tx = tx_indices[p];

    for (int64_t d = 0; d < planes; ++d) {
        A[d * stride] = make_float2(0.0f, 0.0f);
        T[d * stride] = make_float2(0.0f, 0.0f);
    }
    for (int64_t l = 0; l < loops; ++l) {
        const float phase = phase_const * rate * times[l * n_tx + tx];
        float sin_p, cos_p;
        sincosf_reduced(phase, &sin_p, &cos_p);
        for (int64_t d = 0; d < planes; ++d) {
            const float w_re = weight_real[d * loops + l];
            const float w_im = weight_imag[d * loops + l];
            float2 a = A[d * stride];
            a.x += w_re * cos_p - w_im * sin_p;
            a.y += w_re * sin_p + w_im * cos_p;
            A[d * stride] = a;
        }
    }

    // out = contrib * A, so d/d contrib = conj(A) * grad_out, summed over
    // the planes; and T collects what the rate gradient needs.
    for (int64_t s = 0; s < spread; ++s) {
        const int64_t idx = s * paths + p;
        const float c_re = contrib_real[idx];
        const float c_im = contrib_imag[idx];
        const int64_t flat = flat_indices[idx];
        float gc_re = 0.0f, gc_im = 0.0f;
        for (int64_t d = 0; d < planes; ++d) {
            const int64_t out_index = d * base + flat;
            const float go_re = grad_out_real[out_index];
            const float go_im = grad_out_imag[out_index];
            const float2 a = A[d * stride];
            gc_re += go_re * a.x + go_im * a.y;
            gc_im += -go_re * a.y + go_im * a.x;
            float2 t = T[d * stride];
            t.x += go_re * c_re + go_im * c_im;
            t.y += go_re * c_im - go_im * c_re;
            T[d * stride] = t;
        }
        grad_contrib_real[idx] = gc_re;
        grad_contrib_imag[idx] = gc_im;
    }

    // dA[d]/d rate = sum_l W[d,l] * j * phase_const * t(l,tx) * e^{j phi},
    // so grad_rate = sum_d Re(T[d] * dA[d]/d rate)
    //              = sum_l phase_const * t * Re(U_l * j * e^{j phi}),
    // with U_l = sum_d W[d,l] * T[d]. Same ingredients as the 64-plane
    // backward, one linear map further in.
    float rate_accum = 0.0f;
    for (int64_t l = 0; l < loops; ++l) {
        const float t = times[l * n_tx + tx];
        const float phase = phase_const * rate * t;
        float sin_p, cos_p;
        sincosf_reduced(phase, &sin_p, &cos_p);
        float u_re = 0.0f, u_im = 0.0f;
        for (int64_t d = 0; d < planes; ++d) {
            const float w_re = weight_real[d * loops + l];
            const float w_im = weight_imag[d * loops + l];
            const float2 tt = T[d * stride];
            u_re += w_re * tt.x - w_im * tt.y;
            u_im += w_re * tt.y + w_im * tt.x;
        }
        // j * e^{j phi} = (-sin, cos); Re(U * that) = -U.x*sin - U.y*cos
        rate_accum += (-u_re * sin_p - u_im * cos_p) * phase_const * t;
    }
    grad_rates[p] = rate_accum;
}

// ---------------------------------------------------------------------
// Plane-chunked backward. Same arithmetic as
// `doppler_plane_splat_backward_kernel`, but it holds only TILE planes in
// shared memory at a time instead of all of them.
//
// Why: shared memory is `2 * planes * BLOCK * sizeof(float2)` and the launcher
// keeps it under 32 KB, so the block size falls as the plane count rises --
// 128 threads at 13 planes, 64 at 31, and 32 (the floor) at the 63 the RA term
// now reads. Measured per-plane cost tracks it exactly: 54.6 us, 90.8 us,
// 120.2 us per plane. The kernel is occupancy-bound, not work-bound.
//
// Chunking across *blocks* would be the obvious fix and is wrong: `gc` sums
// over every plane and `rate_accum` sums a per-loop reduction over every
// plane, so separate blocks would have to atomicAdd into `grad_contrib`
// (spread x paths) and `grad_rates`. Chunking *inside* one thread costs
// nothing of the sort: both sums are linear in the plane index, so they
// accumulate in registers across chunks and are written once at the end.
//
// The price is the phase. `sincosf_reduced` runs once per (loop, chunk)
// instead of once per loop, i.e. `ceil(planes/TILE)` times over. That buys
// BLOCK back, and at 63 planes it is a 4x occupancy trade for a 4x phase
// recompute.
//
// `spread` is a runtime value, so the per-spread registers are sized by
// MAX_SPREAD and the launcher falls back when it does not fit. The trainer
// renders spread 15.
#define DOPPLER_CHUNK_MAX_SPREAD 16

template <int TILE>
__global__ void doppler_plane_splat_backward_chunked_kernel(
    const float* __restrict__ grad_out_real,
    const float* __restrict__ grad_out_imag,
    const float* __restrict__ contrib_real,
    const float* __restrict__ contrib_imag,
    const int* __restrict__ flat_indices,
    const int64_t* __restrict__ tx_indices,
    const float* __restrict__ path_rates,
    const float* __restrict__ times,
    const float* __restrict__ weight_real,
    const float* __restrict__ weight_imag,
    const float phase_const,
    float* __restrict__ grad_contrib_real,
    float* __restrict__ grad_contrib_imag,
    float* __restrict__ grad_rates,
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base, int64_t planes)
{
    extern __shared__ float2 shared_accum[];
    const int64_t stride = blockDim.x;
    float2* A = shared_accum + threadIdx.x;
    float2* T = shared_accum + (int64_t)TILE * stride + threadIdx.x;

    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= paths) return;
    const float rate = path_rates[p];
    const int64_t tx = tx_indices[p];

    // The spread's inputs are re-read once per chunk otherwise; hoist them.
    float c_re[DOPPLER_CHUNK_MAX_SPREAD], c_im[DOPPLER_CHUNK_MAX_SPREAD];
    int64_t flat[DOPPLER_CHUNK_MAX_SPREAD];
    float gc_re[DOPPLER_CHUNK_MAX_SPREAD], gc_im[DOPPLER_CHUNK_MAX_SPREAD];
    for (int64_t s = 0; s < spread; ++s) {
        const int64_t idx = s * paths + p;
        c_re[s] = contrib_real[idx];
        c_im[s] = contrib_imag[idx];
        flat[s] = flat_indices[idx];
        gc_re[s] = 0.0f;
        gc_im[s] = 0.0f;
    }
    float rate_accum = 0.0f;

    for (int64_t d0 = 0; d0 < planes; d0 += TILE) {
        const int64_t dn = (planes - d0 < (int64_t)TILE) ? (planes - d0) : (int64_t)TILE;
        for (int64_t d = 0; d < dn; ++d) {
            A[d * stride] = make_float2(0.0f, 0.0f);
            T[d * stride] = make_float2(0.0f, 0.0f);
        }
        for (int64_t l = 0; l < loops; ++l) {
            const float phase = phase_const * rate * times[l * n_tx + tx];
            float sin_p, cos_p;
            sincosf_reduced(phase, &sin_p, &cos_p);
            for (int64_t d = 0; d < dn; ++d) {
                const float w_re = weight_real[(d0 + d) * loops + l];
                const float w_im = weight_imag[(d0 + d) * loops + l];
                float2 a = A[d * stride];
                a.x += w_re * cos_p - w_im * sin_p;
                a.y += w_re * sin_p + w_im * cos_p;
                A[d * stride] = a;
            }
        }
        for (int64_t s = 0; s < spread; ++s) {
            for (int64_t d = 0; d < dn; ++d) {
                const int64_t out_index = (d0 + d) * base + flat[s];
                const float go_re = grad_out_real[out_index];
                const float go_im = grad_out_imag[out_index];
                const float2 a = A[d * stride];
                gc_re[s] += go_re * a.x + go_im * a.y;
                gc_im[s] += -go_re * a.y + go_im * a.x;
                float2 t = T[d * stride];
                t.x += go_re * c_re[s] + go_im * c_im[s];
                t.y += go_re * c_im[s] - go_im * c_re[s];
                T[d * stride] = t;
            }
        }
        for (int64_t l = 0; l < loops; ++l) {
            const float t = times[l * n_tx + tx];
            const float phase = phase_const * rate * t;
            float sin_p, cos_p;
            sincosf_reduced(phase, &sin_p, &cos_p);
            float u_re = 0.0f, u_im = 0.0f;
            for (int64_t d = 0; d < dn; ++d) {
                const float w_re = weight_real[(d0 + d) * loops + l];
                const float w_im = weight_imag[(d0 + d) * loops + l];
                const float2 tt = T[d * stride];
                u_re += w_re * tt.x - w_im * tt.y;
                u_im += w_re * tt.y + w_im * tt.x;
            }
            rate_accum += (-u_re * sin_p - u_im * cos_p) * phase_const * t;
        }
    }

    for (int64_t s = 0; s < spread; ++s) {
        const int64_t idx = s * paths + p;
        grad_contrib_real[idx] = gc_re[s];
        grad_contrib_imag[idx] = gc_im[s];
    }
    grad_rates[p] = rate_accum;
}

// The chunked kernel above, with the plane amplitudes READ from the
// forward's `amps` buffer instead of rebuilt. The gather forward already
// computed A[p, d] = sum_l W[d, l] e^{j phi(p, l)} with the same
// accumulation order over `l` this kernel's rebuild loop used, so the loaded
// values are bitwise the rebuilt ones and the whole first per-chunk loop --
// `loops` sincos calls plus loops x TILE FMAs plus the shared-memory
// read-modify-writes, times ceil(planes/TILE) chunks -- disappears. The
// rate-gradient loop keeps its own sincos; it needs the phasor itself, not
// the plane sums.
// TRANSPOSED: the incoming gradient is read from `grad_out_t`, the
// (base, planes) float2 transpose, so a path's (spread, tile) read is TILE
// contiguous float2 per cell instead of TILE loads scattered by `base` (the
// same arithmetic in the same order: bitwise the planes-major result). On
// the cascade the planes-major working set (62 planes x 147k cells x 8 B =
// 73 MB) is the whole L2, and the 4-of-32-byte sectors go to DRAM.
template <int TILE, int SPREAD, bool TRANSPOSED = false>
__global__ void __launch_bounds__(128, 3) doppler_plane_splat_backward_chunked_amps_kernel(
    const float2* __restrict__ forward_amps,  // (paths, planes)
    const float* __restrict__ grad_out_real,
    const float* __restrict__ grad_out_imag,
    const float* __restrict__ contrib_real,
    const float* __restrict__ contrib_imag,
    const int* __restrict__ flat_indices,
    const int64_t* __restrict__ tx_indices,
    const float* __restrict__ path_rates,
    const float* __restrict__ times,
    const float* __restrict__ weight_real,
    const float* __restrict__ weight_imag,
    const float phase_const,
    float* __restrict__ grad_contrib_real,
    float* __restrict__ grad_contrib_imag,
    float* __restrict__ grad_rates,
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base, int64_t planes,
    const int64_t* __restrict__ unit_of_path = nullptr,
    // Per-unit active plane count: planes at or past it are zero-weight pads
    // (their W rows are all zero), so their plane sums are zero and the
    // gradient they carry is zero -- the tiles that would compute them are
    // skipped instead. Bitwise the same result as walking every plane: the
    // skipped terms were exact zeros added to the running sums.
    const int* __restrict__ active_planes = nullptr,
    // Closed form (doppler_closed_form.cuh) when non-null: the rate gradient
    // is sum_d Re(T_d dA_d/drate) with the analytic dA_d, and the 64-loop
    // inner loop below is not run. Needs `unit_of_path`.
    const int* __restrict__ plane_bins = nullptr,
    int windowed = 0, int anchor = 0, const double* __restrict__ tx_times = nullptr,
    const float2* __restrict__ grad_out_t = nullptr)
{
    extern __shared__ float2 shared_accum[];
    const int64_t stride = blockDim.x;
    float2* A = shared_accum + threadIdx.x;
    float2* T = shared_accum + (int64_t)TILE * stride + threadIdx.x;

    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    // Per-unit W when supplied: each frame reads its own
    // Doppler bins out of a (units, planes, loops) matrix.
    const int64_t wbase = (unit_of_path != nullptr && p < paths)
                        ? unit_of_path[p] * planes * loops : 0;
    if (p >= paths) return;
    const int64_t my_planes = (unit_of_path != nullptr && active_planes != nullptr)
                            ? (int64_t)active_planes[unit_of_path[p]] : planes;
    const float rate = path_rates[p];
    const int64_t tx = tx_indices[p];
    const float2* amp_row = forward_amps + p * planes;
    const bool closed = (plane_bins != nullptr);
    ClosedPath cp;
    const int* my_bins = nullptr;
    if (closed) {
        cp = closed_path(rate, tx_times, tx, phase_const, (int)loops);
        my_bins = plane_bins + unit_of_path[p] * planes;
    }
    const int dc = (int)(loops / 2);

    constexpr int SP = (SPREAD > 0) ? SPREAD : DOPPLER_CHUNK_MAX_SPREAD;
    float c_re[SP], c_im[SP];
    int flat[SP];
    float gc_re[SP], gc_im[SP];
    #pragma unroll
    for (int s = 0; s < SP; ++s) {
        if (s >= spread) break;
        const int64_t idx = s * paths + p;
        c_re[s] = contrib_real[idx];
        c_im[s] = contrib_imag[idx];
        flat[s] = flat_indices[idx];
        gc_re[s] = 0.0f;
        gc_im[s] = 0.0f;
    }
    float rate_accum = 0.0f;

    for (int64_t d0 = 0; d0 < my_planes; d0 += TILE) {
        const int64_t dn = (my_planes - d0 < (int64_t)TILE) ? (my_planes - d0) : (int64_t)TILE;
        for (int64_t d = 0; d < dn; ++d) {
            A[d * stride] = amp_row[d0 + d];
            T[d * stride] = make_float2(0.0f, 0.0f);
        }
        #pragma unroll
        for (int s = 0; s < SP; ++s) {
            if (s >= spread) break;
            const float2* row_t = TRANSPOSED ? (grad_out_t + (int64_t)flat[s] * planes + d0) : nullptr;
            for (int64_t d = 0; d < dn; ++d) {
                float go_re, go_im;
                if (TRANSPOSED) {
                    const float2 go = row_t[d];
                    go_re = go.x; go_im = go.y;
                } else {
                    const int64_t out_index = (d0 + d) * base + (int64_t)flat[s];
                    go_re = grad_out_real[out_index];
                    go_im = grad_out_imag[out_index];
                }
                const float2 a = A[d * stride];
                gc_re[s] += go_re * a.x + go_im * a.y;
                gc_im[s] += -go_re * a.y + go_im * a.x;
                float2 t = T[d * stride];
                t.x += go_re * c_re[s] + go_im * c_im[s];
                t.y += go_re * c_im[s] - go_im * c_re[s];
                T[d * stride] = t;
            }
        }
        if (closed) {
            for (int64_t d = 0; d < dn; ++d) {
                const int64_t plane = d0 + d;
                int unshifted = -1;
                if (plane > 0) {
                    const int bin = my_bins[plane];
                    if (bin < 0) continue;
                    unshifted = (bin + dc) % (int)loops;
                }
                float2 Ad, dAd;
                closed_form_plane(cp, (int)loops, unshifted, windowed, anchor, &Ad, &dAd);
                const float2 tt = T[d * stride];
                rate_accum += tt.x * dAd.x - tt.y * dAd.y;
            }
            continue;
        }
        for (int64_t l = 0; l < loops; ++l) {
            const float t = times[l * n_tx + tx];
            const float phase = phase_const * rate * t;
            float sin_p, cos_p;
            sincosf_reduced(phase, &sin_p, &cos_p);
            float u_re = 0.0f, u_im = 0.0f;
            for (int64_t d = 0; d < dn; ++d) {
                const float w_re = weight_real[wbase + (d0 + d) * loops + l];
                const float w_im = weight_imag[wbase + (d0 + d) * loops + l];
                const float2 tt = T[d * stride];
                u_re += w_re * tt.x - w_im * tt.y;
                u_im += w_re * tt.y + w_im * tt.x;
            }
            rate_accum += (-u_re * sin_p - u_im * cos_p) * phase_const * t;
        }
    }

    #pragma unroll
    for (int s = 0; s < SP; ++s) {
        if (s >= spread) break;
        const int64_t idx = s * paths + p;
        grad_contrib_real[idx] = gc_re[s];
        grad_contrib_imag[idx] = gc_im[s];
    }
    grad_rates[p] = rate_accum;
}

// Amps-fed AND transposed-grad: the two independent rewrites combined --
// the forward's plane amplitudes are loaded instead of rebuilt, and the
// incoming gradient is read from its (base, planes) float2 transpose so a
// (spread, tile) read is TILE contiguous float2s. Same accumulation order
// as the baseline chunked kernel in both sums, so the outputs are bitwise
// its outputs.
template <int TILE>
__global__ void doppler_plane_splat_backward_chunked_amps_t_kernel(
    const float2* __restrict__ forward_amps,  // (paths, planes)
    const float2* __restrict__ grad_out,      // (base, planes)
    const float* __restrict__ contrib_real,
    const float* __restrict__ contrib_imag,
    const int* __restrict__ flat_indices,
    const int64_t* __restrict__ tx_indices,
    const float* __restrict__ path_rates,
    const float* __restrict__ times,
    const float* __restrict__ weight_real,
    const float* __restrict__ weight_imag,
    const float phase_const,
    float* __restrict__ grad_contrib_real,
    float* __restrict__ grad_contrib_imag,
    float* __restrict__ grad_rates,
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base, int64_t planes)
{
    extern __shared__ float2 shared_accum[];
    const int64_t stride = blockDim.x;
    float2* A = shared_accum + threadIdx.x;
    float2* T = shared_accum + (int64_t)TILE * stride + threadIdx.x;

    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= paths) return;
    const float rate = path_rates[p];
    const int64_t tx = tx_indices[p];
    const float2* amp_row = forward_amps + p * planes;

    float c_re[DOPPLER_CHUNK_MAX_SPREAD], c_im[DOPPLER_CHUNK_MAX_SPREAD];
    int64_t flat[DOPPLER_CHUNK_MAX_SPREAD];
    float gc_re[DOPPLER_CHUNK_MAX_SPREAD], gc_im[DOPPLER_CHUNK_MAX_SPREAD];
    for (int64_t s = 0; s < spread; ++s) {
        const int64_t idx = s * paths + p;
        c_re[s] = contrib_real[idx];
        c_im[s] = contrib_imag[idx];
        flat[s] = flat_indices[idx];
        gc_re[s] = 0.0f;
        gc_im[s] = 0.0f;
    }
    float rate_accum = 0.0f;

    for (int64_t d0 = 0; d0 < planes; d0 += TILE) {
        const int64_t dn = (planes - d0 < (int64_t)TILE) ? (planes - d0) : (int64_t)TILE;
        for (int64_t d = 0; d < dn; ++d) {
            A[d * stride] = amp_row[d0 + d];
            T[d * stride] = make_float2(0.0f, 0.0f);
        }
        for (int64_t s = 0; s < spread; ++s) {
            const float2* row = grad_out + flat[s] * planes + d0;
            for (int64_t d = 0; d < dn; ++d) {
                const float2 go = row[d];
                const float2 a = A[d * stride];
                gc_re[s] += go.x * a.x + go.y * a.y;
                gc_im[s] += -go.x * a.y + go.y * a.x;
                float2 t = T[d * stride];
                t.x += go.x * c_re[s] + go.y * c_im[s];
                t.y += go.x * c_im[s] - go.y * c_re[s];
                T[d * stride] = t;
            }
        }
        for (int64_t l = 0; l < loops; ++l) {
            const float t = times[l * n_tx + tx];
            const float phase = phase_const * rate * t;
            float sin_p, cos_p;
            sincosf_reduced(phase, &sin_p, &cos_p);
            float u_re = 0.0f, u_im = 0.0f;
            for (int64_t d = 0; d < dn; ++d) {
                const float w_re = weight_real[(d0 + d) * loops + l];
                const float w_im = weight_imag[(d0 + d) * loops + l];
                const float2 tt = T[d * stride];
                u_re += w_re * tt.x - w_im * tt.y;
                u_im += w_re * tt.y + w_im * tt.x;
            }
            rate_accum += (-u_re * sin_p - u_im * cos_p) * phase_const * t;
        }
    }

    for (int64_t s = 0; s < spread; ++s) {
        const int64_t idx = s * paths + p;
        grad_contrib_real[idx] = gc_re[s];
        grad_contrib_imag[idx] = gc_im[s];
    }
    grad_rates[p] = rate_accum;
}

// The chunked kernel above, reading its incoming gradient from the
// (base, planes) float2 transpose instead of (planes, base) planes-major
// pairs. At 63 planes the planes-major read is spread x planes separate
// 8-byte loads scattered by `base` -- 4 useful bytes per 32-byte sector;
// transposed, each (spread, tile) read is TILE contiguous float2s. The
// arithmetic and its order are identical term for term, so the outputs are
// bitwise the chunked kernel's (gated in the suite and by the standalone
// A/B); only where the same values are read from changes.
template <int TILE>
__global__ void doppler_plane_splat_backward_chunked_t_kernel(
    const float2* __restrict__ grad_out,      // (base, planes)
    const float* __restrict__ contrib_real,
    const float* __restrict__ contrib_imag,
    const int* __restrict__ flat_indices,
    const int64_t* __restrict__ tx_indices,
    const float* __restrict__ path_rates,
    const float* __restrict__ times,
    const float* __restrict__ weight_real,
    const float* __restrict__ weight_imag,
    const float phase_const,
    float* __restrict__ grad_contrib_real,
    float* __restrict__ grad_contrib_imag,
    float* __restrict__ grad_rates,
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base, int64_t planes)
{
    extern __shared__ float2 shared_accum[];
    const int64_t stride = blockDim.x;
    float2* A = shared_accum + threadIdx.x;
    float2* T = shared_accum + (int64_t)TILE * stride + threadIdx.x;

    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= paths) return;
    const float rate = path_rates[p];
    const int64_t tx = tx_indices[p];

    float c_re[DOPPLER_CHUNK_MAX_SPREAD], c_im[DOPPLER_CHUNK_MAX_SPREAD];
    int64_t flat[DOPPLER_CHUNK_MAX_SPREAD];
    float gc_re[DOPPLER_CHUNK_MAX_SPREAD], gc_im[DOPPLER_CHUNK_MAX_SPREAD];
    for (int64_t s = 0; s < spread; ++s) {
        const int64_t idx = s * paths + p;
        c_re[s] = contrib_real[idx];
        c_im[s] = contrib_imag[idx];
        flat[s] = flat_indices[idx];
        gc_re[s] = 0.0f;
        gc_im[s] = 0.0f;
    }
    float rate_accum = 0.0f;

    for (int64_t d0 = 0; d0 < planes; d0 += TILE) {
        const int64_t dn = (planes - d0 < (int64_t)TILE) ? (planes - d0) : (int64_t)TILE;
        for (int64_t d = 0; d < dn; ++d) {
            A[d * stride] = make_float2(0.0f, 0.0f);
            T[d * stride] = make_float2(0.0f, 0.0f);
        }
        for (int64_t l = 0; l < loops; ++l) {
            const float phase = phase_const * rate * times[l * n_tx + tx];
            float sin_p, cos_p;
            sincosf_reduced(phase, &sin_p, &cos_p);
            for (int64_t d = 0; d < dn; ++d) {
                const float w_re = weight_real[(d0 + d) * loops + l];
                const float w_im = weight_imag[(d0 + d) * loops + l];
                float2 a = A[d * stride];
                a.x += w_re * cos_p - w_im * sin_p;
                a.y += w_re * sin_p + w_im * cos_p;
                A[d * stride] = a;
            }
        }
        for (int64_t s = 0; s < spread; ++s) {
            const float2* row = grad_out + flat[s] * planes + d0;
            for (int64_t d = 0; d < dn; ++d) {
                const float2 go = row[d];
                const float2 a = A[d * stride];
                gc_re[s] += go.x * a.x + go.y * a.y;
                gc_im[s] += -go.x * a.y + go.y * a.x;
                float2 t = T[d * stride];
                t.x += go.x * c_re[s] + go.y * c_im[s];
                t.y += go.x * c_im[s] - go.y * c_re[s];
                T[d * stride] = t;
            }
        }
        for (int64_t l = 0; l < loops; ++l) {
            const float t = times[l * n_tx + tx];
            const float phase = phase_const * rate * t;
            float sin_p, cos_p;
            sincosf_reduced(phase, &sin_p, &cos_p);
            float u_re = 0.0f, u_im = 0.0f;
            for (int64_t d = 0; d < dn; ++d) {
                const float w_re = weight_real[(d0 + d) * loops + l];
                const float w_im = weight_imag[(d0 + d) * loops + l];
                const float2 tt = T[d * stride];
                u_re += w_re * tt.x - w_im * tt.y;
                u_im += w_re * tt.y + w_im * tt.x;
            }
            rate_accum += (-u_re * sin_p - u_im * cos_p) * phase_const * t;
        }
    }

    for (int64_t s = 0; s < spread; ++s) {
        const int64_t idx = s * paths + p;
        grad_contrib_real[idx] = gc_re[s];
        grad_contrib_imag[idx] = gc_im[s];
    }
    grad_rates[p] = rate_accum;
}

// ---------------------------------------------------------------------
// Follow-on B: K2b's backward, register-tiled over a transposed grad_out.
//
// The kernel below (`doppler_plane_splat_backward_kernel`) keeps its two
// plane accumulators, A[d] and T[d], in shared memory and reads its incoming
// gradient as `grad_out[d * base + flat]` -- so the 14 planes of one output
// cell are 14 *separate* 32-byte sectors, 4 useful bytes each, twice over for
// real and imaginary. Measured, that is the larger half of its 0.940 ms.
//
// Two changes, neither of which touches the arithmetic or its order:
//   * `grad_out` is transposed once into (base, planes) float2, so a cell's
//     planes are 112 contiguous bytes;
//   * A and T live in registers, tiled, exactly as the forward's amplitude
//     pass does since W1.
//
// Both are value-preserving by construction -- same terms, same order -- and
// the whole backward is checked bitwise against the shared-memory version.
// ---------------------------------------------------------------------

__global__ void doppler_plane_grad_transpose_kernel(
    const float* __restrict__ grad_out_real,  // (planes, base)
    const float* __restrict__ grad_out_imag,
    float2* __restrict__ transposed,          // (base, planes)
    int64_t base, int64_t planes)
{
    // One block per 32 cells: reads stay coalesced along `base`, the staged
    // tile is written back contiguously along `planes`.
    extern __shared__ float2 tile[];          // (32, planes)
    const int64_t cell0 = (int64_t)blockIdx.x * 32;
    const int lane = threadIdx.x;
    if (lane < 32) {
        const int64_t cell = cell0 + lane;
        if (cell < base)
            for (int64_t d = 0; d < planes; ++d)
                tile[lane * planes + d] = make_float2(
                    grad_out_real[d * base + cell], grad_out_imag[d * base + cell]);
    }
    __syncthreads();
    const int64_t span = min((int64_t)32, base - cell0) * planes;
    for (int64_t i = threadIdx.x; i < span; i += blockDim.x)
        transposed[cell0 * planes + i] = tile[i];
}

template <int TILE>
__global__ void doppler_plane_splat_backward_tiled_kernel(
    const float2* __restrict__ grad_out,      // (base, planes)
    const float* __restrict__ contrib_real,   // (spread, paths)
    const float* __restrict__ contrib_imag,
    const int* __restrict__ flat_indices, // (spread, paths)
    const int64_t* __restrict__ tx_indices,   // (paths,)
    const float* __restrict__ path_rates,     // (paths,)
    const float* __restrict__ times,          // (loops, n_tx)
    const float* __restrict__ weight_real,    // (planes, loops)
    const float* __restrict__ weight_imag,
    const float phase_const,
    float* __restrict__ grad_contrib_real,    // (spread, paths)
    float* __restrict__ grad_contrib_imag,
    float* __restrict__ grad_rates,           // (paths,)
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base, int64_t planes)
{
    extern __shared__ float2 shared_weight[];  // (planes, loops)
    for (int64_t i = threadIdx.x; i < planes * loops; i += blockDim.x)
        shared_weight[i] = make_float2(weight_real[i], weight_imag[i]);
    __syncthreads();

    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= paths) return;
    const float rate = path_rates[p];
    const int64_t tx = tx_indices[p];

    float2 A[TILE], T[TILE];
#pragma unroll
    for (int j = 0; j < TILE; ++j) {
        A[j] = make_float2(0.0f, 0.0f);
        T[j] = make_float2(0.0f, 0.0f);
    }
    for (int64_t l = 0; l < loops; ++l) {
        const float phase = phase_const * rate * times[l * n_tx + tx];
        float sin_p, cos_p;
        sincosf_reduced(phase, &sin_p, &cos_p);
#pragma unroll
        for (int j = 0; j < TILE; ++j) {
            if (j >= planes) continue;
            const float2 w = shared_weight[j * loops + l];
            A[j].x += w.x * cos_p - w.y * sin_p;
            A[j].y += w.x * sin_p + w.y * cos_p;
        }
    }

    for (int64_t s = 0; s < spread; ++s) {
        const int64_t idx = s * paths + p;
        const float c_re = contrib_real[idx];
        const float c_im = contrib_imag[idx];
        const float2* row = grad_out + flat_indices[idx] * planes;
        float gc_re = 0.0f, gc_im = 0.0f;
#pragma unroll
        for (int j = 0; j < TILE; ++j) {
            if (j >= planes) continue;
            const float2 go = row[j];
            gc_re += go.x * A[j].x + go.y * A[j].y;
            gc_im += -go.x * A[j].y + go.y * A[j].x;
            T[j].x += go.x * c_re + go.y * c_im;
            T[j].y += go.x * c_im - go.y * c_re;
        }
        grad_contrib_real[idx] = gc_re;
        grad_contrib_imag[idx] = gc_im;
    }

    float rate_accum = 0.0f;
    for (int64_t l = 0; l < loops; ++l) {
        const float t = times[l * n_tx + tx];
        const float phase = phase_const * rate * t;
        float sin_p, cos_p;
        sincosf_reduced(phase, &sin_p, &cos_p);
        float u_re = 0.0f, u_im = 0.0f;
#pragma unroll
        for (int j = 0; j < TILE; ++j) {
            if (j >= planes) continue;
            const float2 w = shared_weight[j * loops + l];
            u_re += w.x * T[j].x - w.y * T[j].y;
            u_im += w.x * T[j].y + w.y * T[j].x;
        }
        rate_accum += (-u_re * sin_p - u_im * cos_p) * phase_const * t;
    }
    grad_rates[p] = rate_accum;
}

// ---------------------------------------------------------------------
// E9a (Phase 5 W1): K2b's forward as a gather.
//
// The splat above computes A[d,p] = sum_l W[d,l] e^{j phi(p,l)} in shared
// memory and then *scatters* it, paying 2 * spread * planes global atomicAdds
// per path -- 134 M per training step, better than 97% of that kernel's time
// (0.2 of the step). Inverted, one block owns one output cell
// and accumulates over the (spread, path) pairs that land in it, in registers,
// writing every output element exactly once. No atomics on the output at all.
//
// Nothing here assumes any structure in `flat_indices`: the pairs are binned
// by their destination cell, which is what the scatter reads too. The five
// stages are
//
//   1. amplitudes  A[u,p,d], one thread per (source, path). Same accumulation
//                  order over `l` as the splat, so A is bitwise what the splat
//                  computed; only the summation over paths reorders.
//   2. count       histogram of the spread*paths pairs by destination cell.
//   3. scan        exclusive prefix sum over the `base` cells (one block).
//   4. place       pair -> slot, carrying (contrib, path) as one 16-byte store.
//   5. gather      one block per cell, (sources*planes) x groups threads.
//
// The `sources` axis exists for the evaluation pass: its five velocity
// variants of a chunk share their geometry, their contributions and therefore
// stages 2-4 entirely, and differ only in `path_rates`. One launch over
// (sources x planes) amortizes the binning across all five.
// ---------------------------------------------------------------------

// W staged in shared. `SHARED_W=false` reads it straight from global instead:
// the index depends only on (plane, loop), never on threadIdx, so every lane
// of a warp reads the same address either way -- L1 serves it as a broadcast
// exactly as shared does, and the 31.5 KB stays resident because it is the
// same 31.5 KB on every block of every launch.
//
// Why it matters: the staging costs `planes * loops * sizeof(float2)` of
// DYNAMIC shared per block -- 32,256 B at the RA term's 63 planes -- and that,
// not the register file, is what caps this kernel. Measured from the built
// cubin: REG 101 at BLOCK 128 is 12,928 registers per block, so registers
// allow 5 blocks/SM (20 warps of 48, 42%); 32,256 B allows 3 (12 warps, 26%).
// Dropping the staging removes the cap and the `__syncthreads`.
//
// Bitwise: only the memory space of an operand changes. Same values, same
// order, same FMA sequence. `HUPR_AMPLITUDES_SHARED_W=1` restores the staging.
// How W reaches the inner loop. `W_FULL` stages all of it once, `W_TILE`
// stages only the TILE rows the current tile reads, `W_GLOBAL` reads it from
// L1. See the launcher for which wins and by how much.
#define AMP_W_FULL   0
#define AMP_W_TILE   1
#define AMP_W_GLOBAL 2

// The closed form (doppler_closed_form.cuh): one thread per (path, source),
// no loop over the 64 loops, no W. `plane_bins` is (units, planes) int32,
// the fftshifted bin of every plane (entry 0, the anchor, is ignored; -1 is
// a pad and writes zero); `windowed` and `anchor` are the window and the
// static-anchor kind. Planes at or past the unit's active count are zero.
__global__ void doppler_plane_amplitudes_closed_kernel(
    const int64_t* __restrict__ tx_indices,   // (paths,)
    const float* __restrict__ path_rates,     // (sources, paths)
    const float* __restrict__ times,          // (loops, n_tx)
    const float phase_const,
    float2* __restrict__ amps,                // (sources, paths, planes)
    int64_t paths, int64_t loops, int64_t n_tx, int64_t planes,
    const int64_t* __restrict__ unit_of_path,
    const int* __restrict__ active_planes,
    const int* __restrict__ plane_bins,
    int windowed, int anchor, const double* __restrict__ tx_times)
{
    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t source = blockIdx.y;
    if (p >= paths) return;
    const int64_t unit = unit_of_path[p];
    const int64_t my_planes = (active_planes != nullptr) ? (int64_t)active_planes[unit] : planes;
    const ClosedPath cp = closed_path(path_rates[source * paths + p], tx_times,
                                      tx_indices[p], phase_const, (int)loops);
    float2* out = amps + (source * paths + p) * planes;
    const int* bins = plane_bins + unit * planes;
    const int dc = (int)(loops / 2);
    for (int64_t d = 0; d < planes; ++d) {
        float2 A = make_float2(0.0f, 0.0f), dA;
        if (d < my_planes) {
            int unshifted = -1;
            if (d > 0) {
                const int bin = bins[d];
                if (bin < 0) { out[d] = A; continue; }
                unshifted = (bin + dc) % (int)loops;
            }
            closed_form_plane(cp, (int)loops, unshifted, windowed, anchor, &A, &dA);
        }
        out[d] = A;
    }
}

template <int TILE, int W_MODE>
__global__ void doppler_plane_amplitudes_kernel(
    const int64_t* __restrict__ tx_indices,   // (paths,)
    const float* __restrict__ path_rates,     // (sources, paths)
    const float* __restrict__ times,          // (loops, n_tx)
    const float* __restrict__ weight_real,    // (planes, loops) or (units, planes, loops)
    const float* __restrict__ weight_imag,
    const float phase_const,
    float2* __restrict__ amps,                // (sources, paths, planes)
    int64_t paths, int64_t loops, int64_t n_tx, int64_t planes,
    // nullptr: one W for every path, the historical layout, and the shared
    // staging modes are usable. Non-null: W is (units, planes, loops) and this
    // gives each path its unit, so every frame can supervise its own Doppler
    // bins. Only AMP_W_GLOBAL is valid then -- the staged modes hold one
    // block-wide W, and a block's paths may span two units.
    const int64_t* __restrict__ unit_of_path = nullptr,
    // Per-unit active plane count (needs `unit_of_path`): planes at or past
    // a unit's count are zero-weight pads, written as exact zeros without
    // being accumulated. The block walks tiles up to the widest count among
    // its paths, so a block inside one unit does that unit's work only.
    const int* __restrict__ active_planes = nullptr)
{
    extern __shared__ float2 shared_weight[];
    if (W_MODE == AMP_W_FULL) {
        for (int64_t i = threadIdx.x; i < planes * loops; i += blockDim.x)
            shared_weight[i] = make_float2(weight_real[i], weight_imag[i]);
        __syncthreads();
    }

    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t source = blockIdx.y;
    const bool active = p < paths;
    const float rate = active ? path_rates[source * paths + p] : 0.0f;
    const int64_t tx = active ? tx_indices[p] : 0;
    const int64_t wbase = (unit_of_path != nullptr && active)
                        ? unit_of_path[p] * planes * loops : 0;
    // Does every path in this block share one unit? A block is 128 paths and a
    // unit has thousands, so only the few blocks straddling a boundary say no,
    // and those alone pay the global read.
    __shared__ int block_uniform;
    __shared__ int64_t block_wbase;
    __shared__ int block_planes;
    if (threadIdx.x == 0) {
        block_uniform = 1;
        block_wbase = 0;
        block_planes = (active_planes != nullptr && unit_of_path != nullptr)
                     ? 0 : (int)planes;
        if (unit_of_path != nullptr) {
            const int64_t first = (int64_t)blockIdx.x * blockDim.x;
            const int64_t last = min(first + blockDim.x - 1, paths - 1);
            if (first < paths) {
                block_wbase = unit_of_path[first] * planes * loops;
                block_uniform = (unit_of_path[first] == unit_of_path[last]) ? 1 : 0;
            }
        }
    }
    __syncthreads();
    const int64_t my_planes = (active_planes != nullptr && unit_of_path != nullptr)
                            ? (active ? (int64_t)active_planes[unit_of_path[p]] : 0)
                            : planes;
    if (active_planes != nullptr && unit_of_path != nullptr) {
        atomicMax(&block_planes, (int)my_planes);
        __syncthreads();
    }
    // Staging is only legal when the whole block reads the same W.
    const bool stage_w = (unit_of_path == nullptr) || (block_uniform != 0);
    float2* out = amps + (source * paths + p) * planes;

    // The plane accumulators live in registers, tiled: the splat kept them in
    // shared, which cost a read-modify-write per (loop, plane).
    const int64_t walk_planes = (int64_t)block_planes;
    for (int64_t d0 = 0; d0 < walk_planes; d0 += TILE) {
        const int span = (int)min((int64_t)TILE, planes - d0);
        if (W_MODE == AMP_W_TILE) {
            // Only the rows this tile reads. The first barrier is what makes
            // re-staging safe -- every thread must be done reading the
            // previous tile before its slots are overwritten.
            __syncthreads();
            for (int64_t i = threadIdx.x; i < (int64_t)span * loops;
                 i += blockDim.x)
                shared_weight[i] = make_float2(
                    weight_real[block_wbase + d0 * loops + i],
                    weight_imag[block_wbase + d0 * loops + i]);
            __syncthreads();
        }
        float2 accum[TILE];
#pragma unroll
        for (int j = 0; j < TILE; ++j) accum[j] = make_float2(0.0f, 0.0f);
        for (int64_t l = 0; l < loops; ++l) {
            const float phase = phase_const * rate * times[l * n_tx + tx];
            float sin_p, cos_p;
            sincosf_reduced(phase, &sin_p, &cos_p);
#pragma unroll
            for (int j = 0; j < TILE; ++j) {
                if (j >= span) continue;
                const int64_t wi = (d0 + j) * loops + l;
                const float2 w =
                    (W_MODE == AMP_W_FULL)   ? shared_weight[wi]
                  : (W_MODE == AMP_W_TILE && stage_w)
                                             ? shared_weight[(int64_t)j * loops + l]
                                             : make_float2(weight_real[wbase + wi],
                                                           weight_imag[wbase + wi]);
                accum[j].x += w.x * cos_p - w.y * sin_p;
                accum[j].y += w.x * sin_p + w.y * cos_p;
            }
        }
        if (active) {
#pragma unroll
            for (int j = 0; j < TILE; ++j)
                if (j < span)
                    out[d0 + j] = (d0 + j < my_planes) ? accum[j]
                                                       : make_float2(0.0f, 0.0f);
        }
    }
    // Planes past the block's widest count: pads for every path here.
    if (active)
        for (int64_t d = walk_planes; d < planes; ++d)
            out[d] = make_float2(0.0f, 0.0f);
}

__global__ void doppler_plane_count_kernel(
    const int* __restrict__ flat_indices, int* __restrict__ counts,
    int64_t pairs)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    atomicAdd(&counts[flat_indices[i]], 1);
}

// One block, one tile of `blockDim.x` cells at a time, warp-shuffle scans and
// a single carry between tiles. `base` is 36,864 for a training step and
// 92,160 for an evaluation chunk, so one block is the right shape and the
// result is deterministic by construction.
__global__ void doppler_plane_scan_kernel(
    const int* __restrict__ counts, int* __restrict__ offsets, int base)
{
    __shared__ int warp_totals[32];
    __shared__ int carry;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    for (int start = 0; start < base; start += blockDim.x) {
        const int index = start + threadIdx.x;
        const int value = (index < base) ? counts[index] : 0;
        int scanned = value;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int other = __shfl_up_sync(0xffffffff, scanned, offset);
            if (lane >= offset) scanned += other;
        }
        if (lane == 31) warp_totals[warp] = scanned;
        __syncthreads();
        if (warp == 0) {
            int total = (lane < warps) ? warp_totals[lane] : 0;
            for (int offset = 1; offset < 32; offset <<= 1) {
                const int other = __shfl_up_sync(0xffffffff, total, offset);
                if (lane >= offset) total += other;
            }
            if (lane < warps) warp_totals[lane] = total;
        }
        __syncthreads();
        const int prefix = carry + (warp > 0 ? warp_totals[warp - 1] : 0);
        if (index < base) offsets[index] = prefix + scanned - value;
        __syncthreads();
        if (threadIdx.x == 0) carry += warp_totals[warps - 1];
        __syncthreads();
    }
    if (threadIdx.x == 0) offsets[base] = carry;
}

__global__ void doppler_plane_place_kernel(
    const int* __restrict__ flat_indices, // (spread, paths)
    const float* __restrict__ contrib_real,   // (spread, paths)
    const float* __restrict__ contrib_imag,
    const int* __restrict__ offsets,          // (base + 1)
    int* __restrict__ cursor,                 // (base)
    float4* __restrict__ slots,               // (spread * paths)
    int64_t pairs, int64_t paths)
{
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    const int64_t cell = flat_indices[i];
    const int slot = offsets[cell] + atomicAdd(&cursor[cell], 1);
    // Contribution and path index in one 16-byte store: the destination is
    // scattered, so the sector count is what this pass pays for.
    slots[slot] = make_float4(contrib_real[i], contrib_imag[i],
                              __int_as_float((int)(i % paths)), 0.0f);
}

// W4 (Phase 5): the deterministic binning.
//
// The placement above assigns slots with `atomicAdd(&cursor[cell], 1)`, so the
// order of a cell's contributions -- and therefore the order the gather sums
// them in -- is an arrival draw. That is the last nondeterministic reduction
// this codebase owns. These three kernels replace count + place with a
// counting sort whose slot order is **increasing pair index**, always:
//
//   * `block_count`  — a private histogram per block over its own contiguous
//     chunk of pairs. Counts are order-independent, so atomics are fine here.
//   * `block_scan`   — exclusive scan of those histograms over blocks, per
//     cell, plus the per-cell total the outer scan needs.
//   * `place_ordered` — each block walks its chunk in increasing pair index,
//     warp by warp in order, and `__match_any_sync` ranks the lanes that share
//     a cell by lane id. Block order, warp order and lane order are all
//     ascending in the pair index, so the slot order is exactly it.
//
// The cost is one histogram of (blocks x base) ints, which is why the block
// count is fixed rather than derived from the pair count.
// ---------------------------------------------------------------------

__global__ void doppler_plane_block_count_kernel(
    const int* __restrict__ flat_indices, int* __restrict__ hist,
    int64_t pairs, int64_t base, int64_t chunk)
{
    const int64_t begin = (int64_t)blockIdx.x * chunk;
    const int64_t end = min(begin + chunk, pairs);
    int* mine = hist + (int64_t)blockIdx.x * base;
    for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x)
        atomicAdd(&mine[flat_indices[i]], 1);
}

__global__ void doppler_plane_block_scan_kernel(
    int* __restrict__ hist, int* __restrict__ totals,
    int64_t base, int blocks)
{
    const int64_t cell = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= base) return;
    int running = 0;
    int b = 0;
    // Unrolled by four: the loads are independent of each other and only the
    // accumulator is serial, so issuing four before consuming any of them is
    // what keeps this pass off the memory-latency floor (measured 0.370 ->
    // 0.228 ms/step at 341 blocks).
    for (; b + 4 <= blocks; b += 4) {
        int* e0 = hist + (int64_t)b * base + cell;
        int* e1 = e0 + base;
        int* e2 = e1 + base;
        int* e3 = e2 + base;
        const int v0 = *e0, v1 = *e1, v2 = *e2, v3 = *e3;
        *e0 = running;              running += v0;
        *e1 = running;              running += v1;
        *e2 = running;              running += v2;
        *e3 = running;              running += v3;
    }
    for (; b < blocks; ++b) {
        int* entry = hist + (int64_t)b * base + cell;
        const int value = *entry;
        *entry = running;
        running += value;
    }
    totals[cell] = running;
}

__global__ void doppler_plane_place_ordered_kernel(
    const int* __restrict__ flat_indices,
    const float* __restrict__ contrib_real,
    const float* __restrict__ contrib_imag,
    const int* __restrict__ offsets,          // (base + 1), over cells
    int* __restrict__ cursor,                 // (blocks, base), scanned
    float4* __restrict__ slots,
    int64_t pairs, int64_t paths, int64_t base, int64_t chunk)
{
    const int64_t begin = (int64_t)blockIdx.x * chunk;
    const int64_t end = min(begin + chunk, pairs);
    int* mine = cursor + (int64_t)blockIdx.x * base;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const unsigned below = (1u << lane) - 1u;

    for (int64_t start = begin; start < end; start += blockDim.x) {
        const int64_t i = start + threadIdx.x;
        const bool valid = i < end;
        const int cell = valid ? (int)flat_indices[i] : 0;
        for (int w = 0; w < warps; ++w) {
            if (warp == w) {
                // Every lane of this warp is here, so the full mask is right.
                const unsigned live = __ballot_sync(0xffffffffu, valid);
                const unsigned same = __match_any_sync(0xffffffffu, cell) & live;
                const int leader = __ffs(same) - 1;
                int rank = (valid && lane == leader) ? mine[cell] : 0;
                rank = __shfl_sync(0xffffffffu, rank, leader < 0 ? 0 : leader);
                if (valid) {
                    const int slot = offsets[cell] + rank + __popc(same & below);
                    slots[slot] = make_float4(
                        contrib_real[i], contrib_imag[i],
                        __int_as_float((int)(i % paths)), 0.0f);
                    if (lane == leader) mine[cell] = rank + __popc(same);
                }
            }
            __syncthreads();
        }
    }
}

// blockDim.x = sources * planes, blockDim.y = groups (a power of two). Thread
// (lane, g) accumulates its (source, plane) over the cell's slots g, g+groups,
// ... in registers; a shared reduction over g finishes the cell and writes
// each output element exactly once.
__global__ void doppler_plane_gather_kernel(
    const int* __restrict__ offsets,
    const float4* __restrict__ slots,
    const float2* __restrict__ amps,          // (sources, paths, planes)
    float* __restrict__ out_real,             // (sources * planes * base)
    float* __restrict__ out_imag,
    int64_t base, int64_t paths, int64_t planes,
    // Per-unit active plane count, indexed by `cell / unit_base` (the units
    // are laid out unit-major in `base`). A lane past its unit's count is a
    // pad plane: its amplitudes are exact zeros, so it writes zero and skips
    // the slot walk.
    const int* __restrict__ active_planes = nullptr, int64_t unit_base = 0)
{
    extern __shared__ float2 partial[];
    const int cell = blockIdx.x;
    const int begin = offsets[cell];
    const int end = offsets[cell + 1];
    const int lane = threadIdx.x;            // source * planes + plane
    const int lanes = blockDim.x;
    const int g = threadIdx.y;
    const int groups = blockDim.y;
    const int64_t source = lane / planes;
    const int64_t plane = lane - source * planes;
    const int64_t my_planes = (active_planes != nullptr && unit_base > 0)
                            ? (int64_t)active_planes[cell / unit_base] : planes;
    // (A per-slot reach test for the window route -- skip the slots whose
    // path's Doppler bin is beyond the window -- was measured and removed:
    // its presence in this loop, even compiled out at run time, cost 2.8 ms
    // a step (4.5 -> 7.3 ms) on the cascade, and enabled it cost 4.3.)

    float2 accum = make_float2(0.0f, 0.0f);
    const int walk_end = (plane < my_planes) ? end : begin;
    for (int slot = begin + g; slot < walk_end; slot += groups) {
        const float4 entry = slots[slot];
        const int64_t path = (int64_t)__float_as_int(entry.z);
        const float2 a = amps[(source * paths + path) * planes + plane];
        accum.x += entry.x * a.x - entry.y * a.y;
        accum.y += entry.x * a.y + entry.y * a.x;
    }
    partial[g * lanes + lane] = accum;
    __syncthreads();
    for (int half = groups >> 1; half > 0; half >>= 1) {
        if (g < half) {
            const float2 other = partial[(g + half) * lanes + lane];
            partial[g * lanes + lane].x += other.x;
            partial[g * lanes + lane].y += other.y;
        }
        __syncthreads();
    }
    if (g == 0) {
        const int64_t index = (int64_t)lane * base + cell;
        out_real[index] = partial[lane].x;
        out_imag[index] = partial[lane].y;
    }
}

// ---------------------------------------------------------------------
// The 15-tap Doppler PSF route (docs/TRAINING_SPEED_PLAN_2.md 3.5).
//
// With the Hann window a path's Doppler response is compact -- 1.0e-3 of
// its peak at 7 bins from the path's fractional bin -- so, exactly as the
// range PSF is truncated to 15 bins, a path deposits into the WINDOW of
// `taps` bins around its own Doppler bin only. `amps_w` is (paths, 1 + taps)
// float2: entry 0 the static anchor (every path), entries 1..taps the
// closed-form amplitude at unshifted bins n - half .. n + half, zero where
// the unit does not render that bin, entries 1+taps and 2+taps the folded
// mean-removal term alone at bins 1 and L-1 (zero when those bins lie in
// the window, where the window entry carries it). `bin_to_plane` is (units, loops + 1)
// int32: the unit's plane index (1..) for each unshifted bin, -1 when not
// rendered (column `loops` is the pads' dummy). `n_of_path` is the path's
// integer Doppler bin, for the gather. One velocity source only.
// ---------------------------------------------------------------------

__global__ void doppler_plane_amplitudes_window_kernel(
    const int64_t* __restrict__ tx_indices,   // (paths,)
    const float* __restrict__ path_rates,     // (paths,)
    const float phase_const,
    float2* __restrict__ amps_w,              // (paths, 3 + taps)
    int* __restrict__ n_of_path,              // (paths,)
    int64_t paths, int64_t loops, int64_t planes,
    const int64_t* __restrict__ unit_of_path,
    const int* __restrict__ active_planes,
    const int* __restrict__ bin_to_plane,
    int windowed, int anchor, const double* __restrict__ tx_times, int taps)
{
    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= paths) return;
    const int64_t unit = unit_of_path[p];
    const int64_t my_planes = (active_planes != nullptr) ? (int64_t)active_planes[unit] : planes;
    const ClosedPath cp = closed_path(path_rates[p], tx_times, tx_indices[p], phase_const, (int)loops);
    n_of_path[p] = cp.n;
    const int* table = bin_to_plane + unit * (loops + 1);
    // Compact (paths, 3 + taps) output: entry 0 the anchor, 1..taps the
    // window, taps+1 and taps+2 the fold-only bins 1 and L-1; an exact zero
    // where the path deposits nothing (the gather and the backward skip a
    // zero). The dense (paths, planes) layout this replaced was 972 MB a
    // step at 64 planes, a 1 ms memset and a 4.3 ms gather of 90% zeros.
    const int W = 3 + taps;
    float2* out = amps_w + p * W;
    float2 A, dA;
    closed_form_plane(cp, (int)loops, -1, windowed, anchor, &A, &dA);
    out[0] = A;
    const int half = taps / 2;
    const int L = (int)loops;
    for (int k = 0; k < taps; ++k) {
        int bin = (cp.n - half + k) % L;
        if (bin < 0) bin += L;
        const int plane = table[bin];
        if (plane < 0 || plane >= my_planes) { out[1 + k] = make_float2(0.0f, 0.0f); continue; }
        closed_form_plane(cp, L, bin, windowed, anchor, &A, &dA);
        out[1 + k] = A;
    }
    // the folded mean-removal term at bins 1 and L-1 for every path: it is
    // not compact (every path leaks its D(-nu) there), so outside the window
    // it is deposited on its own
    for (int f = 0; f < 2; ++f) {
        const int bin = (f == 0) ? 1 : L - 1;
        int rel = (bin - cp.n) % L; if (rel < 0) rel += L; if (rel >= L / 2) rel -= L;
        const int plane = table[bin];
        if (!windowed || plane < 0 || plane >= my_planes || (rel >= -half && rel <= half)) {
            out[1 + taps + f] = make_float2(0.0f, 0.0f);
            continue;
        }
        closed_form_plane(cp, L, bin, windowed, anchor, &A, &dA, 1);
        out[1 + taps + f] = A;
    }
}

// The bin of window entry `w` (1..taps), fold entry (taps+1, taps+2) or -1 (anchor).
__device__ __forceinline__ int window_entry_bin(int w, int n, int half, int taps, int L) {
    if (w == 0) return -1;
    if (w <= taps) { int bin = (n - half + (w - 1)) % L; if (bin < 0) bin += L; return bin; }
    return (w == taps + 1) ? 1 : L - 1;
}

// The window route's gather: one block per output cell, four warps, each
// warp walking every fourth slot of the cell with its lanes over the
// path's 3 + taps entries. The lanes of one slot land on distinct planes
// (distinct window bins; the fold entries are exact zeros where a window
// bin already carries the fold), so a warp's read-modify-write of its own
// shared accumulators needs no atomics, and the four warps' accumulators
// are summed in a fixed order: the cell's sum is a function of its slot
// order alone, exactly as the dense gather's was. Per slot the traffic is
// 16 + 8 (3 + taps) bytes against 16 + 8 planes, and one warp instruction
// stream against two warps of plane lanes.
__global__ void doppler_plane_gather_taps_kernel(
    const int* __restrict__ offsets,
    const float4* __restrict__ slots,
    const float2* __restrict__ amps_w,        // (paths, 3 + taps)
    const int* __restrict__ n_of_path,
    float* __restrict__ out_real,             // (planes * base)
    float* __restrict__ out_imag,
    int64_t base, int64_t planes, int64_t loops,
    const int* __restrict__ bin_to_plane, int64_t unit_base, int taps)
{
    extern __shared__ float2 acc[];           // (warps, planes)
    const int cell = blockIdx.x;
    const int warps = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < warps * planes; i += blockDim.x) acc[i] = make_float2(0.0f, 0.0f);
    __syncthreads();
    const int begin = offsets[cell];
    const int end = offsets[cell + 1];
    const int64_t unit = cell / unit_base;
    const int* table = bin_to_plane + unit * (loops + 1);
    const int half = taps / 2;
    const int L = (int)loops;
    const int W = 3 + taps;
    float2* mine = acc + (int64_t)warp * planes;
    for (int slot = begin + warp; slot < end; slot += warps) {
        const float4 entry = slots[slot];                 // one address: a broadcast
        const int path = __float_as_int(entry.z);
        if (lane < W) {
            const float2 v = amps_w[(int64_t)path * W + lane];
            if (v.x != 0.0f || v.y != 0.0f) {
                const int plane = (lane == 0) ? 0
                                : table[window_entry_bin(lane, n_of_path[path], half, taps, L)];
                float2 a = mine[plane];
                a.x += entry.x * v.x - entry.y * v.y;
                a.y += entry.x * v.y + entry.y * v.x;
                mine[plane] = a;
            }
        }
        __syncwarp();
    }
    __syncthreads();
    for (int d = threadIdx.x; d < planes; d += blockDim.x) {
        float2 sum = make_float2(0.0f, 0.0f);
        for (int w = 0; w < warps; ++w) {
            sum.x += acc[(int64_t)w * planes + d].x;
            sum.y += acc[(int64_t)w * planes + d].y;
        }
        out_real[(int64_t)d * base + cell] = sum.x;
        out_imag[(int64_t)d * base + cell] = sum.y;
    }
}

// ---------------------------------------------------------------------
// The sorted Doppler route (docs/TRAINING_SPEED_PLAN_2.md 3.1/3.4 for the
// person): the batch's paths in (unit, channel, range bin) order, as the
// scene's static map is (`_scene_frames_geometry`). A cell (unit, channel,
// range bin) then owns one or two contiguous runs of that order, its
// contributions are synthesised from the per-entry record (carrier,
// fractional and integer range bin) and the range PSF lerp, and the path's
// Doppler amplitudes (its compact window row, in entry order) are read by
// the lanes of the cell and applied to every plane of the window. No
// binning per step, no (spread, paths) contribution arrays, and the 15
// cells that read one path's amplitude row are neighbours in the same
// block. The order itself is fixed per frame (cached in the trainer's row
// geometry) and assembled per batch by a scatter.
// ---------------------------------------------------------------------

constexpr int SORTED_BINS = 32;             // range bins per block: one lane per cell
constexpr int SORTED_WARPS = 8;             // warps per block, each walking every eighth entry
constexpr int SORTED_MAX_PLANES = 64;
constexpr int SORTED_ACC_STRIDE = SORTED_MAX_PLANES + 1;   // 65: lanes l and l+16 share a bank, no worse

// One block per (unit, channel, 32 consecutive range bins), one lane per
// cell, eight warps splitting the block's entries. The block walks the
// union of its cells' entries ONCE (the paths whose 15-tap range window
// touches the block's bins, one or two runs of the sorted order): per
// entry the record and the path's window row are loaded once for the
// warp, each lane synthesises its own tap's contribution (its bin minus
// the path's bin) and adds the window into its own accumulator row with
// shared-memory atomics (the warps' order is not fixed, as the default
// binning's slot order is not; `--deterministic on` keeps the slot route).
// Measured on the way: a cell-per-half-warp layout visited every entry once
// per cell (15x the visits) and was L2-bound at 2.5 ms; one warp per block
// with a fixed order was residency-bound (16.6 KB of accumulators per
// warp: six warps an SM) at 6 ms.
__global__ void __launch_bounds__(SORTED_WARPS * 32) doppler_plane_sorted_forward_kernel(
    const int4* __restrict__ runs,            // (blocks,) begin1, end1, begin2, end2 into the entries
    const float4* __restrict__ record,        // (entries,) carrier re, im, n_frac, (n_dopp << 16 | n_floor)
    const float2* __restrict__ amps_e,        // (entries, W) the window row, entry order
    const int* __restrict__ bin_to_plane,     // (units, L + 1)
    const float4* __restrict__ table4,        // (spread, n_grid) packed PSF table
    float* __restrict__ out_real,             // (planes, cells)
    float* __restrict__ out_imag,
    int64_t cells, int64_t unit_base, int K, int half_range, int n_grid,
    int planes, int L, int taps)
{
    __shared__ float2 acc[SORTED_BINS * SORTED_ACC_STRIDE];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int blocks_per_group = K / SORTED_BINS;
    const int64_t group = blockIdx.x / blocks_per_group;            // (unit, channel)
    const int rbin0 = (int)(blockIdx.x % blocks_per_group) * SORTED_BINS;
    const int rbin = rbin0 + lane;
    const int64_t cell = group * K + rbin;
    for (int i = threadIdx.x; i < SORTED_BINS * SORTED_ACC_STRIDE; i += blockDim.x)
        acc[i] = make_float2(0.0f, 0.0f);
    __syncthreads();
    const int4 r = runs[blockIdx.x];
    const int* table = bin_to_plane + (cell / unit_base) * (L + 1);
    const int half_d = taps / 2;
    const int W = 3 + taps;
    const int spread = 2 * half_range + 1;
    float2* mine = acc + lane * SORTED_ACC_STRIDE;
    const int n1 = max(r.y - r.x, 0), n2 = max(r.w - r.z, 0);
    const int total = n1 + n2;
    for (int i = warp; i < total; i += SORTED_WARPS) {
        const int e = (i < n1) ? r.x + i : r.z + (i - n1);
        const float4 rec = record[e];                          // one address: a broadcast
        float2 a_mine = make_float2(0.0f, 0.0f);
        if (lane < W) a_mine = amps_e[(int64_t)e * W + lane];
        const int packed = __float_as_int(rec.w);
        const int n_floor = packed & 0xffff;
        const int n_dopp = packed >> 16;
        int p_mine = -1;
        if (lane < W && (a_mine.x != 0.0f || a_mine.y != 0.0f))
            p_mine = (lane == 0) ? 0 : table[window_entry_bin(lane, n_dopp, half_d, taps, L)];
        int dn = rbin - n_floor;
        if (dn > half_range) dn -= K;
        if (dn < -half_range) dn += K;
        const int s = dn + half_range;
        const bool valid = (s >= 0 && s < spread);
        float c_re = 0.0f, c_im = 0.0f;
        if (valid) {
            float pr, pi;
            psf_lerp(table4, s, rec.z, n_grid, &pr, &pi);
            c_re = rec.x * pr - rec.y * pi;
            c_im = rec.x * pi + rec.y * pr;
        }
        // lanes 0..W-1 hold the window row and its planes; every lane takes
        // them by shuffle (broadcast loads in this loop measured 3x slower)
        for (int w = 0; w < W; ++w) {
            const float ax = __shfl_sync(0xffffffffu, a_mine.x, w);
            const float ay = __shfl_sync(0xffffffffu, a_mine.y, w);
            const int plane = __shfl_sync(0xffffffffu, p_mine, w);
            if (valid && plane >= 0) {
                atomicAdd(&mine[plane].x, c_re * ax - c_im * ay);
                atomicAdd(&mine[plane].y, c_re * ay + c_im * ax);
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < SORTED_BINS * planes; i += blockDim.x) {
        const int l = i % SORTED_BINS, d = i / SORTED_BINS;      // lanes over cells: coalesced stores
        const float2 v = acc[l * SORTED_ACC_STRIDE + d];
        out_real[(int64_t)d * cells + group * K + rbin0 + l] = v.x;
        out_imag[(int64_t)d * cells + group * K + rbin0 + l] = v.y;
    }
}

// The window amplitudes in entry order: `path_of_entry` maps the entry to
// its path (rates, TX and unit are the path's).
__global__ void doppler_plane_amplitudes_sorted_kernel(
    const int* __restrict__ path_of_entry,    // (entries,)
    const int64_t* __restrict__ tx_indices,   // (paths,)
    const float* __restrict__ path_rates,     // (paths,)
    const float phase_const,
    float2* __restrict__ amps_e,              // (entries, 3 + taps)
    float4* __restrict__ record,              // (entries,) .w: n_floor in, (n_dopp << 16 | n_floor) out
    int64_t entries, int64_t loops, int64_t planes,
    const int64_t* __restrict__ unit_of_path,
    const int* __restrict__ active_planes,
    const int* __restrict__ bin_to_plane,
    int windowed, int anchor, const double* __restrict__ tx_times, int taps)
{
    const int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= entries) return;
    const int64_t p = path_of_entry[e];
    const int64_t unit = unit_of_path[p];
    const int64_t my_planes = (active_planes != nullptr) ? (int64_t)active_planes[unit] : planes;
    const ClosedPath cp = closed_path(path_rates[p], tx_times, tx_indices[p], phase_const, (int)loops);
    // the path's integer Doppler bin rides in the record's fourth field
    // with the range bin, so the forward reads one word for both
    record[e].w = __int_as_float((cp.n << 16) | (__float_as_int(record[e].w) & 0xffff));
    const int* table = bin_to_plane + unit * (loops + 1);
    const int W = 3 + taps;
    float2* out = amps_e + e * W;
    float2 A, dA;
    closed_form_plane(cp, (int)loops, -1, windowed, anchor, &A, &dA);
    out[0] = A;
    const int half = taps / 2;
    const int L = (int)loops;
    for (int k = 0; k < taps; ++k) {
        int bin = (cp.n - half + k) % L;
        if (bin < 0) bin += L;
        const int plane = table[bin];
        if (plane < 0 || plane >= my_planes) { out[1 + k] = make_float2(0.0f, 0.0f); continue; }
        closed_form_plane(cp, L, bin, windowed, anchor, &A, &dA);
        out[1 + k] = A;
    }
    for (int f = 0; f < 2; ++f) {
        const int bin = (f == 0) ? 1 : L - 1;
        int rel = (bin - cp.n) % L; if (rel < 0) rel += L; if (rel >= L / 2) rel -= L;
        const int plane = table[bin];
        if (!windowed || plane < 0 || plane >= my_planes || (rel >= -half && rel <= half)) {
            out[1 + taps + f] = make_float2(0.0f, 0.0f);
            continue;
        }
        closed_form_plane(cp, L, bin, windowed, anchor, &A, &dA, 1);
        out[1 + taps + f] = A;
    }
}

// The backward, one thread per entry: the carrier gradient (through the
// synthesised contributions) and the rate gradient (through the closed-form
// amplitudes), from the (cells, planes) transpose of the incoming gradient.
// Neighbouring entries are neighbouring range bins of one channel, so their
// cells share cache lines.
__global__ void __launch_bounds__(128, 3) doppler_plane_sorted_backward_kernel(
    const float2* __restrict__ grad_out_t,    // (cells, planes)
    const int* __restrict__ path_of_entry,    // (entries,)
    const float4* __restrict__ record,        // (entries,)
    const float2* __restrict__ amps_e,        // (entries, W)
    const int64_t* __restrict__ tx_indices,   // (paths,)
    const float* __restrict__ path_rates,     // (paths,)
    const int64_t* __restrict__ unit_of_path, // (paths,)
    const int* __restrict__ active_planes,
    const int* __restrict__ bin_to_plane,
    const float4* __restrict__ table4,
    const float phase_const,
    float2* __restrict__ grad_carrier,        // (paths,)
    float* __restrict__ grad_rates,           // (paths,)
    int64_t entries, int64_t unit_base, int channels_per_point, int spread,
    int K, int n_grid, int planes, int L, int taps, int windowed, int anchor,
    const double* __restrict__ tx_times)
{
    const int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= entries) return;
    const int64_t p = path_of_entry[e];
    const int64_t unit = unit_of_path[p];
    const int64_t my_planes = (active_planes != nullptr) ? (int64_t)active_planes[unit] : planes;
    const int channel = (int)(p % channels_per_point);
    const ClosedPath cp = closed_path(path_rates[p], tx_times, tx_indices[p], phase_const, L);
    const int* table = bin_to_plane + unit * (L + 1);
    const float4 rec = record[e];
    const int n_floor = __float_as_int(rec.w) & 0xffff;
    const int half_d = taps / 2;
    const int half_r = spread / 2;
    const int W = 3 + taps;
    // Register arrays: every loop over them is unrolled to the compile-time
    // bound with an early exit at W, so nothing lands in local memory.
    constexpr int MAXW = 18;
    int plane_of[MAXW];
    float2 a_w[MAXW], T[MAXW];
    plane_of[0] = 0;
    #pragma unroll
    for (int w = 1; w < MAXW; ++w) {
        if (w >= W) break;
        const int bin = window_entry_bin(w, cp.n, half_d, taps, L);
        int plane = table[bin];
        if (w > taps) {
            int rel = (bin - cp.n) % L; if (rel < 0) rel += L; if (rel >= L / 2) rel -= L;
            if (!windowed || (rel >= -half_d && rel <= half_d)) plane = -1;
        }
        plane_of[w] = (plane < 0 || plane >= my_planes) ? -1 : plane;
    }
    #pragma unroll
    for (int w = 0; w < MAXW; ++w) {
        if (w >= W) break;
        a_w[w] = amps_e[e * W + w]; T[w] = make_float2(0.0f, 0.0f);
    }
    float2 gc = make_float2(0.0f, 0.0f);
    const int64_t cell_base = unit * unit_base + (int64_t)channel * K;
    for (int s = 0; s < spread; ++s) {
        int rbin = n_floor + s - half_r; rbin %= K; if (rbin < 0) rbin += K;
        const float2* go_row = grad_out_t + (cell_base + rbin) * planes;
        float pr, pi;
        psf_lerp(table4, s, rec.z, n_grid, &pr, &pi);
        const float c_re = rec.x * pr - rec.y * pi;
        const float c_im = rec.x * pi + rec.y * pr;
        float2 gcs = make_float2(0.0f, 0.0f);       // the gradient into this tap's contribution
        #pragma unroll
        for (int w = 0; w < MAXW; ++w) {
            if (w >= W) break;
            const int plane = plane_of[w];
            if (plane < 0) continue;
            const float2 go = go_row[plane];
            const float2 aw = a_w[w];
            gcs.x += go.x * aw.x + go.y * aw.y;
            gcs.y += -go.x * aw.y + go.y * aw.x;
            T[w].x += go.x * c_re + go.y * c_im;
            T[w].y += go.x * c_im - go.y * c_re;
        }
        gc.x += gcs.x * pr + gcs.y * pi;
        gc.y += -gcs.x * pi + gcs.y * pr;
    }
    float rate_accum = 0.0f;
    #pragma unroll
    for (int w = 0; w < MAXW; ++w) {
        if (w >= W) break;
        if (plane_of[w] < 0) continue;
        const int bin = window_entry_bin(w, cp.n, half_d, taps, L);
        float2 A, dA;
        closed_form_plane(cp, L, bin, windowed, anchor, &A, &dA, w > taps ? 1 : 0);
        rate_accum += T[w].x * dA.x - T[w].y * dA.y;
    }
    grad_carrier[p] = gc;
    grad_rates[p] = rate_accum;
}

// The backward over the window: per path, the 1 + taps planes it deposited
// into, read from the (base, planes) transpose of the incoming gradient.
template <int SPREAD>
__global__ void __launch_bounds__(128, 3) doppler_plane_splat_backward_window_kernel(
    const float2* __restrict__ amps_w,        // (paths, 3 + taps)
    const float2* __restrict__ grad_out_t,    // (base, planes)
    const float* __restrict__ contrib_real,
    const float* __restrict__ contrib_imag,
    const int* __restrict__ flat_indices,
    const int64_t* __restrict__ tx_indices,
    const float* __restrict__ path_rates,
    const float phase_const,
    float* __restrict__ grad_contrib_real,
    float* __restrict__ grad_contrib_imag,
    float* __restrict__ grad_rates,
    int64_t spread, int64_t paths, int64_t loops, int64_t planes,
    const int64_t* __restrict__ unit_of_path,
    const int* __restrict__ active_planes,
    const int* __restrict__ bin_to_plane,
    int windowed, int anchor, const double* __restrict__ tx_times, int taps)
{
    const int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= paths) return;
    const int64_t unit = unit_of_path[p];
    const int64_t my_planes = (active_planes != nullptr) ? (int64_t)active_planes[unit] : planes;
    const ClosedPath cp = closed_path(path_rates[p], tx_times, tx_indices[p], phase_const, (int)loops);
    const int* table = bin_to_plane + unit * (loops + 1);
    const float2* a = amps_w + p * (3 + taps);   // compact (paths, 3 + taps)
    constexpr int SP = (SPREAD > 0) ? SPREAD : DOPPLER_CHUNK_MAX_SPREAD;
    constexpr int MAXW = 18;                  // 3 + taps, taps <= 15
    float c_re[SP], c_im[SP];
    int flat[SP];
    float gc_re[SP], gc_im[SP];
    #pragma unroll
    for (int s = 0; s < SP; ++s) {
        if (s >= spread) break;
        const int64_t idx = s * paths + p;
        c_re[s] = contrib_real[idx]; c_im[s] = contrib_imag[idx];
        flat[s] = flat_indices[idx];
        gc_re[s] = 0.0f; gc_im[s] = 0.0f;
    }
    const int half = taps / 2;
    const int L = (int)loops;
    int plane_of[MAXW];
    plane_of[0] = 0;
    for (int w = 1; w <= taps + 2; ++w) {
        const int bin = window_entry_bin(w, cp.n, half, taps, L);
        int plane = table[bin];
        if (w > taps) {
            // the fold-only entries: only outside the window
            int rel = (bin - cp.n) % L; if (rel < 0) rel += L; if (rel >= L / 2) rel -= L;
            if (!windowed || (rel >= -half && rel <= half)) plane = -1;
        }
        plane_of[w] = (plane < 0 || plane >= my_planes) ? -1 : plane;
    }
    float rate_accum = 0.0f;
    for (int w = 0; w <= taps + 2; ++w) {
        const int plane = plane_of[w];
        if (plane < 0) continue;
        const float2 aw = a[w];
        float2 T = make_float2(0.0f, 0.0f);
        #pragma unroll
        for (int s = 0; s < SP; ++s) {
            if (s >= spread) break;
            const float2 go = grad_out_t[(int64_t)flat[s] * planes + plane];
            gc_re[s] += go.x * aw.x + go.y * aw.y;
            gc_im[s] += -go.x * aw.y + go.y * aw.x;
            T.x += go.x * c_re[s] + go.y * c_im[s];
            T.y += go.x * c_im[s] - go.y * c_re[s];
        }
        const int bin = window_entry_bin(w, cp.n, half, taps, L);
        float2 A, dA;
        closed_form_plane(cp, L, bin, windowed, anchor, &A, &dA, w > taps ? 1 : 0);
        rate_accum += T.x * dA.x - T.y * dA.y;
    }
    #pragma unroll
    for (int s = 0; s < SP; ++s) {
        if (s >= spread) break;
        const int64_t idx = s * paths + p;
        grad_contrib_real[idx] = gc_re[s];
        grad_contrib_imag[idx] = gc_im[s];
    }
    grad_rates[p] = rate_accum;
}

// ---------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------

void launch_static_gain_splat_forward(
    const float* contrib_real, const float* contrib_imag,
    const int* flat_indices, const int64_t* point_indices,
    const float* gains, float* out_real, float* out_imag,
    int64_t spread, int64_t paths, cudaStream_t stream)
{
    const int64_t total = spread * paths;
    constexpr int BLOCK = 256;
    const int grid = (int)((total + BLOCK - 1) / BLOCK);
    if (grid <= 0) return;
    static_gain_splat_forward_kernel<<<grid, BLOCK, 0, stream>>>(
        contrib_real, contrib_imag, flat_indices, point_indices, gains,
        out_real, out_imag, spread, paths);
}

void launch_static_gain_splat_backward(
    const float* grad_out_real, const float* grad_out_imag,
    const float* contrib_real, const float* contrib_imag,
    const int* flat_indices, const int64_t* point_indices,
    const float* gains, float* grad_gains,
    float* grad_contrib_real, float* grad_contrib_imag,
    int64_t spread, int64_t paths, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    const int grid = (int)((paths + BLOCK - 1) / BLOCK);
    if (grid <= 0) return;
    static_gain_splat_backward_kernel<<<grid, BLOCK, 0, stream>>>(
        grad_out_real, grad_out_imag, contrib_real, contrib_imag,
        flat_indices, point_indices, gains, grad_gains,
        grad_contrib_real, grad_contrib_imag, spread, paths);
}

void launch_static_scene_frames_forward(
    const int* runs, const int* entry, const float* carrier,
    const float* table4, const float* gains_t,
    float* out, int64_t base, int K, int half, int n_grid, int frames,
    cudaStream_t stream)
{
    if (base <= 0) return;
    constexpr int BLOCK = 256;
    const int64_t threads = base * SCENE_LANES;
    const int grid = (int)((threads + BLOCK - 1) / BLOCK);
    static_scene_frames_forward_kernel<<<grid, BLOCK, 0, stream>>>(
        reinterpret_cast<const int4*>(runs), reinterpret_cast<const int4*>(entry),
        reinterpret_cast<const float2*>(carrier), reinterpret_cast<const float4*>(table4),
        gains_t, reinterpret_cast<float2*>(out), base, K, half, n_grid, frames);
}

void launch_static_scene_frames_backward(
    const float* grad_out_t, const int* entry, const float* table4,
    const float* gains_t, float* grad_carrier,
    int64_t paths, int K, int half, int n_grid, int frames, cudaStream_t stream)
{
    if (paths <= 0) return;
    constexpr int BLOCK = 256;
    const int grid = (int)((paths + BLOCK - 1) / BLOCK);
    static_scene_frames_backward_kernel<<<grid, BLOCK, 0, stream>>>(
        reinterpret_cast<const float2*>(grad_out_t), reinterpret_cast<const int4*>(entry),
        reinterpret_cast<const float4*>(table4), gains_t,
        reinterpret_cast<float2*>(grad_carrier), paths, K, half, n_grid, frames);
}

void launch_doppler_plane_sorted_forward(
    const int* runs, float* record, const int* path_of_entry,
    const int64_t* tx_indices, const float* path_rates, float phase_const,
    float* amps_e, const int64_t* unit_of_path,
    const int* active_planes, const int* bin_to_plane, int windowed, int anchor,
    const double* tx_times, int taps, const float* table4,
    float* out_real, float* out_imag,
    int64_t entries, int64_t cells, int64_t unit_base, int K, int spread,
    int n_grid, int planes, int L, cudaStream_t stream)
{
    if (entries <= 0 || cells <= 0) return;
    {
        constexpr int BLOCK = 128;
        const int grid = (int)((entries + BLOCK - 1) / BLOCK);
        doppler_plane_amplitudes_sorted_kernel<<<grid, BLOCK, 0, stream>>>(
            path_of_entry, tx_indices, path_rates, phase_const,
            reinterpret_cast<float2*>(amps_e), reinterpret_cast<float4*>(record),
            entries, L, planes, unit_of_path, active_planes, bin_to_plane,
            windowed, anchor, tx_times, taps);
    }
    {
        // one warp per (unit, channel, 32 range bins): `runs` is per block
        const int grid = (int)(cells / SORTED_BINS);
        doppler_plane_sorted_forward_kernel<<<grid, SORTED_WARPS * 32, 0, stream>>>(
            reinterpret_cast<const int4*>(runs), reinterpret_cast<const float4*>(record),
            reinterpret_cast<const float2*>(amps_e), bin_to_plane,
            reinterpret_cast<const float4*>(table4), out_real, out_imag,
            cells, unit_base, K, spread / 2, n_grid, planes, L, taps);
    }
}

void launch_doppler_plane_sorted_backward(
    const float* grad_out_real, const float* grad_out_imag, float* grad_scratch,
    const int* path_of_entry, const float* record, const float* amps_e,
    const int64_t* tx_indices, const float* path_rates,
    const int64_t* unit_of_path, const int* active_planes, const int* bin_to_plane,
    const float* table4, float phase_const,
    float* grad_carrier, float* grad_rates,
    int64_t entries, int64_t cells, int64_t unit_base, int channels_per_point,
    int spread, int K, int n_grid, int planes, int L, int taps, int windowed,
    int anchor, const double* tx_times, cudaStream_t stream)
{
    if (entries <= 0) return;
    auto* transposed = reinterpret_cast<float2*>(grad_scratch);
    {
        constexpr int TBLOCK = 128;
        const int tgrid = (int)((cells + 31) / 32);
        const size_t tshared = (size_t)32 * planes * sizeof(float2);
        doppler_plane_grad_transpose_kernel<<<tgrid, TBLOCK, tshared, stream>>>(
            grad_out_real, grad_out_imag, transposed, cells, planes);
    }
    constexpr int BLOCK = 128;
    const int grid = (int)((entries + BLOCK - 1) / BLOCK);
    doppler_plane_sorted_backward_kernel<<<grid, BLOCK, 0, stream>>>(
        transposed, path_of_entry, reinterpret_cast<const float4*>(record),
        reinterpret_cast<const float2*>(amps_e), tx_indices, path_rates,
        unit_of_path, active_planes, bin_to_plane, reinterpret_cast<const float4*>(table4),
        phase_const, reinterpret_cast<float2*>(grad_carrier), grad_rates,
        entries, unit_base, channels_per_point, spread, K, n_grid, planes, L,
        taps, windowed, anchor, tx_times);
}

void launch_static_gain_map_forward(
    const int* cell_offsets, const float* slot_contrib, const int* slot_point,
    const float* gains, float* out_real, float* out_imag,
    int64_t base, cudaStream_t stream)
{
    if (base <= 0) return;
    constexpr int BLOCK = 128;
    const size_t shared = (size_t)BLOCK * sizeof(float2);
    static_gain_map_forward_kernel<<<(unsigned)base, BLOCK, shared, stream>>>(
        cell_offsets, reinterpret_cast<const float2*>(slot_contrib), slot_point,
        gains, out_real, out_imag);
}

void launch_static_gain_map_backward(
    const float* grad_out_real, const float* grad_out_imag,
    const float* contrib_real, const float* contrib_imag,
    const int* flat_indices, const int* point_offsets,
    const int* point_paths, const float* gains, float* grad_gains,
    float* grad_contrib_real, float* grad_contrib_imag,
    int64_t spread, int64_t paths, int64_t n_points, cudaStream_t stream)
{
    if (n_points <= 0) return;
    constexpr int BLOCK = 256;                 // 8 warps, i.e. 8 points
    const int grid = (int)((n_points * 32 + BLOCK - 1) / BLOCK);
    static_gain_map_backward_kernel<<<grid, BLOCK, 0, stream>>>(
        grad_out_real, grad_out_imag, contrib_real, contrib_imag, flat_indices,
        point_offsets, point_paths, gains, grad_gains,
        grad_contrib_real, grad_contrib_imag, spread, paths, n_points);
}

void launch_slow_time_splat_forward(
    const float* contrib_real, const float* contrib_imag,
    const int* flat_indices, const int64_t* tx_indices,
    const float* path_rates, const float* times, float phase_const,
    float* out_real, float* out_imag,
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base, cudaStream_t stream)
{
    constexpr int BLOCK = 128;
    const int grid = (int)((paths + BLOCK - 1) / BLOCK);
    if (grid <= 0) return;
    slow_time_splat_forward_kernel<<<grid, BLOCK, 0, stream>>>(
        contrib_real, contrib_imag, flat_indices, tx_indices, path_rates,
        times, phase_const, out_real, out_imag,
        spread, paths, loops, n_tx, base);
}

void launch_slow_time_splat_backward(
    const float* grad_out_real, const float* grad_out_imag,
    const float* contrib_real, const float* contrib_imag,
    const int* flat_indices, const int64_t* tx_indices,
    const float* path_rates, const float* times, float phase_const,
    float* grad_contrib_real, float* grad_contrib_imag, float* grad_rates,
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base, cudaStream_t stream)
{
    constexpr int BLOCK = 128;
    const int grid = (int)((paths + BLOCK - 1) / BLOCK);
    if (grid <= 0) return;
    slow_time_splat_backward_kernel<<<grid, BLOCK, 0, stream>>>(
        grad_out_real, grad_out_imag, contrib_real, contrib_imag,
        flat_indices, tx_indices, path_rates, times, phase_const,
        grad_contrib_real, grad_contrib_imag, grad_rates,
        spread, paths, loops, n_tx, base);
}

// Shared memory is `slots * planes * BLOCK * sizeof(float2)`; pick the
// largest power-of-two block that keeps it inside 32 KB, so the launcher
// works for any plane count up to 64 (the dense case) as well as the 14
// the trainer asks for.
static int doppler_block_size(int64_t planes, int slots)
{
    for (int block = 256; block > 32; block >>= 1) {
        if ((size_t)slots * planes * block * sizeof(float2) <= 32768) return block;
    }
    return 32;
}

void launch_doppler_plane_splat_forward(
    const float* contrib_real, const float* contrib_imag,
    const int* flat_indices, const int64_t* tx_indices,
    const float* path_rates, const float* times,
    const float* weight_real, const float* weight_imag, float phase_const,
    float* out_real, float* out_imag,
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base, int64_t planes, cudaStream_t stream)
{
    const int block = doppler_block_size(planes, 1);
    const int grid = (int)((paths + block - 1) / block);
    if (grid <= 0) return;
    const size_t shared = (size_t)planes * block * sizeof(float2);
    doppler_plane_splat_forward_kernel<<<grid, block, shared, stream>>>(
        contrib_real, contrib_imag, flat_indices, tx_indices, path_rates,
        times, weight_real, weight_imag, phase_const, out_real, out_imag,
        spread, paths, loops, n_tx, base, planes);
}

// Follow-on B: `grad_scratch` is (base, planes) float2 when the caller wants
// the tiled path, null when it wants the original shared-memory one. The tiled
// kernel holds `planes` complex accumulators twice over in registers, so it is
// selected only up to 16 planes -- the trainer renders 14.
static constexpr int DOPPLER_BACKWARD_TILE = 32;

void launch_doppler_plane_splat_backward(
    const float* grad_out_real, const float* grad_out_imag,
    const float* contrib_real, const float* contrib_imag,
    const int* flat_indices, const int64_t* tx_indices,
    const float* path_rates, const float* times,
    const float* weight_real, const float* weight_imag, float phase_const,
    float* grad_contrib_real, float* grad_contrib_imag, float* grad_rates,
    float* grad_scratch, const float* forward_amps,
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base, int64_t planes, cudaStream_t stream,
    const int64_t* unit_of_path, const int* active_planes,
    const int* plane_bins, int windowed, int anchor, const double* tx_times,
    const int* bin_to_plane, int taps)
{
    if (bin_to_plane != nullptr && taps > 0 && unit_of_path != nullptr) {
        // The window route: the transpose of the incoming gradient is
        // required (the binding allocates it), then one kernel per path.
        constexpr int BLOCK = 128;
        const int grid = (int)((paths + BLOCK - 1) / BLOCK);
        if (grid <= 0) return;
        auto* transposed = reinterpret_cast<float2*>(grad_scratch);
        {
            constexpr int TBLOCK = 128;
            const int tgrid = (int)((base + 31) / 32);
            const size_t tshared = (size_t)32 * planes * sizeof(float2);
            doppler_plane_grad_transpose_kernel<<<tgrid, TBLOCK, tshared, stream>>>(
                grad_out_real, grad_out_imag, transposed, base, planes);
        }
#define WIN_BW(SP)                                                            \
        doppler_plane_splat_backward_window_kernel<SP><<<grid, BLOCK, 0, stream>>>( \
            reinterpret_cast<const float2*>(forward_amps), transposed,        \
            contrib_real, contrib_imag, flat_indices, tx_indices, path_rates, \
            phase_const, grad_contrib_real, grad_contrib_imag, grad_rates,    \
            spread, paths, loops, planes, unit_of_path, active_planes,        \
            bin_to_plane, windowed, anchor, tx_times, taps)
        switch ((int)spread) {
            case 5:  WIN_BW(5);  break;
            case 9:  WIN_BW(9);  break;
            case 15: WIN_BW(15); break;
            case 16: WIN_BW(16); break;
            default: WIN_BW(0);  break;
        }
#undef WIN_BW
        return;
    }
    if (unit_of_path != nullptr) {
        // Per-unit W has exactly ONE route: the amps-fed chunked backward.
        // The other variants either stage a single block-wide W in shared
        // memory (tiled) or exist only as measured A/B alternatives, and a
        // per-unit call reaching one of them would read the wrong frame's
        // Doppler rows and be silently wrong rather than fail. Everything
        // this route needs is asserted, not assumed.
        // Preconditions are enforced in the binding, where they can raise;
        // by here they hold.
        constexpr int TILE = 16;
        constexpr int BLOCK = 128;
        const int grid = (int)((paths + BLOCK - 1) / BLOCK);
        if (grid <= 0) return;
        const size_t shared = (size_t)2 * TILE * BLOCK * sizeof(float2);
        // Transposed reads whenever the scratch exists (the binding
        // allocates it above 16 planes): one transpose pass, then every
        // (spread, tile) read is contiguous. HUPR_DOPPLER_AMPS_T=0 forces
        // the planes-major reads for A/B.
        static const int unit_transposed_forced = [] {
            const char* v = getenv("HUPR_DOPPLER_AMPS_T");
            return v ? ((v[0] == '1') ? 1 : 0) : -1;
        }();
        const bool unit_transposed = (grad_scratch != nullptr)
            && (unit_transposed_forced != 0);
        const float2* grad_t = nullptr;
        if (unit_transposed) {
            auto* transposed = reinterpret_cast<float2*>(grad_scratch);
            constexpr int TBLOCK = 128;
            const int tgrid = (int)((base + 31) / 32);
            const size_t tshared = (size_t)32 * planes * sizeof(float2);
            doppler_plane_grad_transpose_kernel<<<tgrid, TBLOCK, tshared, stream>>>(
                grad_out_real, grad_out_imag, transposed, base, planes);
            grad_t = transposed;
        }
#define AMPS_BW_T(SP, TR)                                                     \
        doppler_plane_splat_backward_chunked_amps_kernel<TILE, SP, TR>        \
            <<<grid, BLOCK, shared, stream>>>(                                \
            reinterpret_cast<const float2*>(forward_amps),                    \
            grad_out_real, grad_out_imag, contrib_real, contrib_imag,         \
            flat_indices, tx_indices, path_rates, times,                      \
            weight_real, weight_imag, phase_const,                            \
            grad_contrib_real, grad_contrib_imag, grad_rates,                 \
            spread, paths, loops, n_tx, base, planes, unit_of_path,           \
            active_planes, plane_bins, windowed, anchor, tx_times, grad_t)
#define AMPS_BW(SP)                                                           \
        do { if (unit_transposed) AMPS_BW_T(SP, true); else AMPS_BW_T(SP, false); } while (0)
        // The five per-thread arrays are indexed by the spread loop, and a
        // dynamically indexed array cannot be register-resident -- it goes to
        // local memory whatever the register budget, which is where the
        // measured 416 B/thread spill came from. Instantiating the spreads
        // that actually occur (`psf_spread` is 15 by default; its ablation
        // axis is {5,9,15,21,25} and 21/25 exceed the chunk maximum anyway)
        // unrolls the loops and puts them in registers. SP=0 keeps the
        // runtime-`spread` form for anything else.
        switch ((int)spread) {
            case 5:  AMPS_BW(5);  break;
            case 9:  AMPS_BW(9);  break;
            case 15: AMPS_BW(15); break;
            case 16: AMPS_BW(16); break;
            default: AMPS_BW(0);  break;
        }
#undef AMPS_BW
#undef AMPS_BW_T
        return;
    }
    if (grad_scratch != nullptr && planes <= DOPPLER_BACKWARD_TILE) {
        auto* transposed = reinterpret_cast<float2*>(grad_scratch);
        {
            constexpr int BLOCK = 128;
            const int grid = (int)((base + 31) / 32);
            const size_t shared = (size_t)32 * planes * sizeof(float2);
            doppler_plane_grad_transpose_kernel<<<grid, BLOCK, shared, stream>>>(
                grad_out_real, grad_out_imag, transposed, base, planes);
        }
        constexpr int BLOCK = 128;
        const int grid = (int)((paths + BLOCK - 1) / BLOCK);
        if (grid <= 0) return;
        const size_t shared = (size_t)planes * loops * sizeof(float2);
        doppler_plane_splat_backward_tiled_kernel<DOPPLER_BACKWARD_TILE>
            <<<grid, BLOCK, shared, stream>>>(
            transposed, contrib_real, contrib_imag, flat_indices, tx_indices,
            path_rates, times, weight_real, weight_imag, phase_const,
            grad_contrib_real, grad_contrib_imag, grad_rates,
            spread, paths, loops, n_tx, base, planes);
        return;
    }
    // Plane-chunked path: it only pays off once the plane count has driven
    // the block below 128, and it needs the spread to fit its registers.
    // `HUPR_DOPPLER_CHUNKED=0` forces the original kernel for A/B.
    static const bool chunked_enabled = [] {
        const char* v = getenv("HUPR_DOPPLER_CHUNKED");
        return !(v && v[0] == '0');
    }();
    if (chunked_enabled && planes > 16 &&
        spread <= DOPPLER_CHUNK_MAX_SPREAD) {
        // Tile/block trade at the same 32 KB shared budget: TILE 32 at
        // BLOCK 64 halves the chunk count (and so the per-(path, loop)
        // sincos recompute) at half the resident threads per SM. Measured on
        // the canonical fb-12 step it is a REGRESSION -- 29.9 against 27.0
        // ms/step -- the same occupancy trade the 99 KB shared-memory
        // experiment lost (this file, doppler_block_size). The default stays
        // 16; `HUPR_DOPPLER_CHUNK_TILE=32` keeps the wide tile runnable for
        // A/B on other shapes.
        static const int chunk_tile = [] {
            const char* v = getenv("HUPR_DOPPLER_CHUNK_TILE");
            return v ? atoi(v) : 16;
        }();
        if (chunk_tile >= 32 && planes > 32) {
            constexpr int TILE = 32;
            constexpr int BLOCK = 64;                  // 2*32*64*8 = 32 KB
            const int grid = (int)((paths + BLOCK - 1) / BLOCK);
            if (grid <= 0) return;
            const size_t shared = (size_t)2 * TILE * BLOCK * sizeof(float2);
            doppler_plane_splat_backward_chunked_kernel<TILE>
                <<<grid, BLOCK, shared, stream>>>(
                grad_out_real, grad_out_imag, contrib_real, contrib_imag,
                flat_indices, tx_indices, path_rates, times,
                weight_real, weight_imag, phase_const,
                grad_contrib_real, grad_contrib_imag, grad_rates,
                spread, paths, loops, n_tx, base, planes);
            return;
        }
        constexpr int TILE = 16;
        constexpr int BLOCK = 128;                     // 2*16*128*8 = 32 KB
        const int grid = (int)((paths + BLOCK - 1) / BLOCK);
        if (grid <= 0) return;
        const size_t shared = (size_t)2 * TILE * BLOCK * sizeof(float2);
        // Amps-fed variant: the forward's plane amplitudes are loaded
        // instead of rebuilt -- bitwise the same values, since the
        // amplitudes kernel accumulates over the loop axis in the same
        // order the rebuild loop did (verified bitwise in the standalone
        // A/B). `HUPR_DOPPLER_AMPS_BACKWARD=0` forces the rebuild.
        static const bool amps_backward = [] {
            const char* v = getenv("HUPR_DOPPLER_AMPS_BACKWARD");
            return !(v && v[0] == '0');
        }();
        // The scattered planes-major grad reads have a working set of
        // planes x base x 8 bytes -- 74 MB at frame-batch 24 against Ada's
        // 72 MB L2, and past it the 4-of-32-byte sectors go to DRAM (the
        // measured cliff: 42 -> 76 ms/step from fb 24 to fb 30). The
        // transposed layout keeps every fetched line fully used, so it is
        // the default wherever the working set does not comfortably fit L2;
        // below that the transpose pass is pure overhead (measured at fb 12:
        // 7.65 vs 7.19 ms). HUPR_DOPPLER_AMPS_T=0/1 forces either form.
        static const int amps_transposed_forced = [] {
            const char* v = getenv("HUPR_DOPPLER_AMPS_T");
            return v ? ((v[0] == '1') ? 1 : 0) : -1;
        }();
        // 80 MB: frame-batch 24's 75.5 MB working set still caches well
        // (measured: planes-major 41.7 vs transposed 42.2 ms/step), so the
        // transposed form only takes over above it.
        const bool amps_transposed =
            (amps_transposed_forced >= 0)
                ? (amps_transposed_forced == 1)
                : (planes * base * (int64_t)sizeof(float2) > (80 << 20));
        if (amps_backward && forward_amps != nullptr) {
            if (amps_transposed && grad_scratch != nullptr) {
                auto* transposed = reinterpret_cast<float2*>(grad_scratch);
                {
                    constexpr int TBLOCK = 128;
                    const int tgrid = (int)((base + 31) / 32);
                    const size_t tshared = (size_t)32 * planes * sizeof(float2);
                    doppler_plane_grad_transpose_kernel
                        <<<tgrid, TBLOCK, tshared, stream>>>(
                        grad_out_real, grad_out_imag, transposed, base, planes);
                }
                doppler_plane_splat_backward_chunked_amps_t_kernel<TILE>
                    <<<grid, BLOCK, shared, stream>>>(
                    reinterpret_cast<const float2*>(forward_amps),
                    transposed, contrib_real, contrib_imag,
                    flat_indices, tx_indices, path_rates, times,
                    weight_real, weight_imag, phase_const,
                    grad_contrib_real, grad_contrib_imag, grad_rates,
                    spread, paths, loops, n_tx, base, planes);
                return;
            }
            doppler_plane_splat_backward_chunked_amps_kernel<TILE, 0>
                <<<grid, BLOCK, shared, stream>>>(
                reinterpret_cast<const float2*>(forward_amps),
                grad_out_real, grad_out_imag, contrib_real, contrib_imag,
                flat_indices, tx_indices, path_rates, times,
                weight_real, weight_imag, phase_const,
                grad_contrib_real, grad_contrib_imag, grad_rates,
                spread, paths, loops, n_tx, base, planes);
            return;
        }
        // Transposed-read variant: same kernel, same accumulation order,
        // but the incoming gradient is transposed once to (base, planes)
        // float2 so a (spread, tile) read is TILE contiguous float2s
        // instead of TILE loads scattered by `base`. Measured on the
        // canonical fb-12 step it is a REGRESSION -- 7.65 against 7.19 ms
        // for the kernel alone, plus the transpose pass: the scattered
        // reads were already L2-resident (76 pairs share a cell on
        // average). Off by default; `HUPR_DOPPLER_CHUNK_TRANSPOSED=1`
        // keeps it runnable for A/B on other shapes (outputs verified
        // bitwise identical to the planes-major kernel).
        static const bool chunk_transposed = [] {
            const char* v = getenv("HUPR_DOPPLER_CHUNK_TRANSPOSED");
            return v && v[0] == '1';
        }();
        if (chunk_transposed && grad_scratch != nullptr) {
            auto* transposed = reinterpret_cast<float2*>(grad_scratch);
            {
                constexpr int TBLOCK = 128;
                const int tgrid = (int)((base + 31) / 32);
                const size_t tshared = (size_t)32 * planes * sizeof(float2);
                doppler_plane_grad_transpose_kernel
                    <<<tgrid, TBLOCK, tshared, stream>>>(
                    grad_out_real, grad_out_imag, transposed, base, planes);
            }
            doppler_plane_splat_backward_chunked_t_kernel<TILE>
                <<<grid, BLOCK, shared, stream>>>(
                transposed, contrib_real, contrib_imag,
                flat_indices, tx_indices, path_rates, times,
                weight_real, weight_imag, phase_const,
                grad_contrib_real, grad_contrib_imag, grad_rates,
                spread, paths, loops, n_tx, base, planes);
            return;
        }
        doppler_plane_splat_backward_chunked_kernel<TILE>
            <<<grid, BLOCK, shared, stream>>>(
            grad_out_real, grad_out_imag, contrib_real, contrib_imag,
            flat_indices, tx_indices, path_rates, times,
            weight_real, weight_imag, phase_const,
            grad_contrib_real, grad_contrib_imag, grad_rates,
            spread, paths, loops, n_tx, base, planes);
        return;
    }
    const int block = doppler_block_size(planes, 2);
    const int grid = (int)((paths + block - 1) / block);
    if (grid <= 0) return;
    const size_t shared = (size_t)2 * planes * block * sizeof(float2);
    doppler_plane_splat_backward_kernel<<<grid, block, shared, stream>>>(
        grad_out_real, grad_out_imag, contrib_real, contrib_imag,
        flat_indices, tx_indices, path_rates, times,
        weight_real, weight_imag, phase_const,
        grad_contrib_real, grad_contrib_imag, grad_rates,
        spread, paths, loops, n_tx, base, planes);
}

// Number of slot groups a gather block runs: a power of two, capped so the
// block stays inside 1024 threads. `lanes` is 14 for a training step and
// sources*14 for a fused evaluation chunk.
// Cells are processed one block each, `lanes` = sources*planes wide and
// `groups` deep. The heuristic fills the block to 1024 threads, which at the
// RA term's 64 lanes is groups 16 -- and at REG 40 that is 40,960 registers a
// block, i.e. ONE block per SM, 32 warps of 48. A narrower block resides
// better: groups 8 is 512 threads, 3 blocks/SM, 48 warps.
//
// It is faster, and it is NOT a default. Measured, seq 102, 14 epochs, mean of
// 7-14, three repeats each (spread <= 0.002):
//
//   groups 16 (default)   1.7421 s/epoch
//   groups  8             1.7306 s/epoch   -0.66%, 5.7 s per 500 epochs
//   groups  4             1.734
//   groups  2             1.760
//
// The reason it stays off: `groups` is the stride of this kernel's own
// reduction -- group g sums slots begin+g, begin+g+groups, ... and then a tree
// reduction folds the groups -- so changing it REASSOCIATES the sum. It is a
// different fixed order, not a bitwise-identical one. Four `verify-refactor`
// step probes put it inside the run's own spread (16-vs-8 objective
// 0.0-6.489e-08 against a 16-vs-16 self-spread of 1.298e-07, worst gradient
// 2.878e-05 against 1.439e-05-2.878e-05), so the project's gate accepts it --
// but 0.66% is not a reason to move a reduction order in a pipeline whose
// equivalence story rests on them. `HUPR_GATHER_GROUPS=8` turns it on.
static int doppler_gather_groups(int lanes)
{
    static const int forced = [] {
        const char* v = getenv("HUPR_GATHER_GROUPS");
        return v ? atoi(v) : 0;
    }();
    if (forced > 0) {
        int groups = 1;
        while (groups < forced && (groups * 2) * lanes <= 1024) groups <<= 1;
        return groups;
    }
    int groups = 1;
    while (groups < 32 && (groups * 2) * lanes <= 1024) groups <<= 1;
    return groups;
}

void launch_doppler_plane_gather_forward(
    const float* contrib_real, const float* contrib_imag,
    const int* flat_indices, const int64_t* tx_indices,
    const float* path_rates, const float* times,
    const float* weight_real, const float* weight_imag, float phase_const,
    float* amps, int* counts, int* offsets, int* cursor, float* slots,
    float* out_real, float* out_imag,
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base, int64_t planes, int64_t sources, int cursor_blocks,
    cudaStream_t stream, const int64_t* unit_of_path,
    const int* active_planes, int64_t unit_base,
    const int* plane_bins, int windowed, int anchor, const double* tx_times,
    const int* bin_to_plane, int taps, int* n_of_path)
{
    if (paths <= 0 || base <= 0) return;
    const int64_t pairs = spread * paths;
    const bool window_route = (bin_to_plane != nullptr && taps > 0 && unit_of_path != nullptr);

    if (window_route) {
        constexpr int BLOCK = 128;
        const dim3 grid((unsigned)((paths + BLOCK - 1) / BLOCK), 1);
        doppler_plane_amplitudes_window_kernel<<<grid, BLOCK, 0, stream>>>(
            tx_indices, path_rates, phase_const, reinterpret_cast<float2*>(amps),
            n_of_path, paths, loops, planes, unit_of_path, active_planes,
            bin_to_plane, windowed, anchor, tx_times, taps);
    } else if (plane_bins != nullptr && unit_of_path != nullptr) {
        constexpr int BLOCK = 128;
        const dim3 grid((unsigned)((paths + BLOCK - 1) / BLOCK), (unsigned)sources);
        doppler_plane_amplitudes_closed_kernel<<<grid, BLOCK, 0, stream>>>(
            tx_indices, path_rates, times, phase_const,
            reinterpret_cast<float2*>(amps), paths, loops, n_tx, planes,
            unit_of_path, active_planes, plane_bins, windowed, anchor, tx_times);
    } else {
        constexpr int BLOCK = 128;
        const dim3 grid((unsigned)((paths + BLOCK - 1) / BLOCK), (unsigned)sources);
        const size_t shared = (size_t)planes * loops * sizeof(float2);
        // The register tile bounds how often the per-(path, loop) sincos is
        // recomputed: ceil(planes/TILE) times over. At the RA term's 64
        // planes the <16> specialization pays it 4x, so wider tiles are
        // instantiated and picked by the plane count. TILE=32 keeps the
        // accumulators in registers (64 per thread); TILE=64 was measured
        // slower (spills). `HUPR_AMPLITUDES_TILE=16` forces the narrow tile
        // for A/B (precedent: HUPR_DOPPLER_CHUNKED).
        static const int forced_tile = [] {
            const char* v = getenv("HUPR_AMPLITUDES_TILE");
            return v ? atoi(v) : 0;
        }();
        const int tile = forced_tile ? forced_tile
                                     : (planes > 16 ? 32 : 16);
        // How W reaches the inner loop. It is (planes, loops) complex, read
        // at an index that never depends on threadIdx -- a pure broadcast.
        // `HUPR_AMPLITUDES_W=full|tile|global` overrides the default.
        //
        // Staging ALL of it costs planes*loops*8 = 32,256 B of dynamic shared
        // per block at the RA term's 63 planes, and that -- not the register
        // file -- is what caps the kernel. From the built cubin, BLOCK 128:
        //
        //   mode    REG   shared/block   blocks/SM   warps of 48   occupancy
        //   full    101       32,256 B       3           12          25%
        //   tile    (see the launcher's table)
        //   global  125            0 B       4           16          33%
        //
        // Registers alone would allow 5 blocks (20 warps, 42%). `global`
        // removes the shared cap but the compiler spends the freed budget on
        // registers and the L1 path costs more instructions, so it loses.
        // `tile` stages only the TILE rows the current tile reads --
        // TILE*loops*8, i.e. 16 KB at TILE 32 and 8 KB at TILE 16 -- which
        // keeps the shared read AND lifts the cap, at two extra barriers per
        // tile. Measured, seq 102, 14 epochs, mean of 7-14, s/epoch: see the
        // ledger entry R033 table.
        static const int w_mode_env = [] {
            const char* v = getenv("HUPR_AMPLITUDES_W");
            if (!v) return AMP_W_TILE;
            if (v[0] == 'f') return AMP_W_FULL;
            if (v[0] == 'g') return AMP_W_GLOBAL;
            return AMP_W_TILE;
        }();
        // Per-unit W keeps the staged tile: the kernel stages the block's own
        // unit and falls back to the global read only for a block that
        // straddles a unit boundary. AMP_W_FULL still cannot -- it stages the
        // whole W once, before the per-block unit is known.
        const int w_mode = (unit_of_path != nullptr && w_mode_env == AMP_W_FULL)
                         ? AMP_W_GLOBAL : w_mode_env;
        const size_t w_shared =
            (w_mode == AMP_W_FULL) ? shared
          : (w_mode == AMP_W_TILE) ? (size_t)tile * loops * sizeof(float2)
                                   : 0;
#define AMP_LAUNCH(T, M)                                                      \
        doppler_plane_amplitudes_kernel<T, M><<<grid, BLOCK, w_shared, stream>>>( \
            tx_indices, path_rates, times, weight_real, weight_imag,          \
            phase_const, reinterpret_cast<float2*>(amps),                     \
            paths, loops, n_tx, planes, unit_of_path,                         \
            (unit_of_path != nullptr) ? active_planes : nullptr)
        if (tile >= 32 && planes > 16) {
            if (w_mode == AMP_W_FULL)        AMP_LAUNCH(32, AMP_W_FULL);
            else if (w_mode == AMP_W_GLOBAL) AMP_LAUNCH(32, AMP_W_GLOBAL);
            else                             AMP_LAUNCH(32, AMP_W_TILE);
        } else {
            if (w_mode == AMP_W_FULL)        AMP_LAUNCH(16, AMP_W_FULL);
            else if (w_mode == AMP_W_GLOBAL) AMP_LAUNCH(16, AMP_W_GLOBAL);
            else                             AMP_LAUNCH(16, AMP_W_TILE);
        }
#undef AMP_LAUNCH
    }
    if (cursor_blocks > 0) {
        // W4: deterministic binning. `cursor` is (cursor_blocks, base).
        //
        // The placement runs **one warp per block**: a block's warps have to
        // be serialized against each other (they share the block's cursor
        // row), so a wider block only idles lanes. Measured on a training
        // step, 256 threads per block cost 0.661 ms against 0.221 ms for 32.
        const int64_t chunk = (pairs + cursor_blocks - 1) / cursor_blocks;
        doppler_plane_block_count_kernel<<<cursor_blocks, 256, 0, stream>>>(
            flat_indices, cursor, pairs, base, chunk);
        doppler_plane_block_scan_kernel<<<
            (unsigned)((base + 255) / 256), 256, 0, stream>>>(
            cursor, counts, base, cursor_blocks);
        doppler_plane_scan_kernel<<<1, 1024, 0, stream>>>(
            counts, offsets, (int)base);
        doppler_plane_place_ordered_kernel<<<cursor_blocks, 32, 0, stream>>>(
            flat_indices, contrib_real, contrib_imag, offsets, cursor,
            reinterpret_cast<float4*>(slots), pairs, paths, base, chunk);
    } else {
        constexpr int BLOCK = 256;
        const int grid = (int)((pairs + BLOCK - 1) / BLOCK);
        doppler_plane_count_kernel<<<grid, BLOCK, 0, stream>>>(
            flat_indices, counts, pairs);
        doppler_plane_scan_kernel<<<1, 1024, 0, stream>>>(
            counts, offsets, (int)base);
        doppler_plane_place_kernel<<<grid, BLOCK, 0, stream>>>(
            flat_indices, contrib_real, contrib_imag, offsets, cursor,
            reinterpret_cast<float4*>(slots), pairs, paths);
    }
    if (window_route) {
        constexpr int BLOCK = 128;
        const size_t shared = (size_t)(BLOCK / 32) * planes * sizeof(float2);
        doppler_plane_gather_taps_kernel<<<(unsigned)base, BLOCK, shared, stream>>>(
            offsets, reinterpret_cast<const float4*>(slots),
            reinterpret_cast<const float2*>(amps), n_of_path, out_real, out_imag,
            base, planes, loops, bin_to_plane, unit_base, taps);
        return;
    }
    {
        const int lanes = (int)(sources * planes);
        const int groups = doppler_gather_groups(lanes);
        const dim3 block((unsigned)lanes, (unsigned)groups);
        const size_t shared = (size_t)lanes * groups * sizeof(float2);
        doppler_plane_gather_kernel<<<(unsigned)base, block, shared, stream>>>(
            offsets, reinterpret_cast<const float4*>(slots),
            reinterpret_cast<const float2*>(amps), out_real, out_imag,
            base, paths, planes,
            (unit_of_path != nullptr) ? active_planes : nullptr, unit_base);
    }
}

// ---------------------------------------------------------------------------
// The elevation term's plane part, pointwise chain fused (docs/RUNTIME_STEP_20MS_PLAN.md A1).
//
// The chain this replaces, as `rae_terms_of` writes it in eager PyTorch, per cell of the
// (units, planes, rows, angle, elevation) store:
//
//     mag  = |spectrum|                                  ~10 passes over 42.3 M cells a step,
//     s    = mag^2 + floor * factor                      each one reading and writing 169 MB,
//     pred = sqrt(max(s, 1e-30))                          measured at 14.03 ms of elementwise
//     t    = (float)target_f16 * rms[unit]                plus 1.41 ms of abs and its sgn
//     out  = (pred - t)^2                                 backward
//
// BITWISE, and two things make that true rather than approximate:
//
//   1. Every operation uses an explicit round-to-nearest intrinsic (`__fmul_rn`, `__fadd_rn`,
//      `__fsub_rn`, `__fdiv_rn`). nvcc contracts `a * b + c` into a single-rounding FMA by
//      default; eager PyTorch performs one operation per kernel, so every intermediate is
//      rounded at the kernel boundary. A contracted kernel rounds once where eager rounds
//      twice and is NOT bitwise it -- measured: torch.compile on this same chain deviates at
//      1.3e-7..2.6e-7 on 20-40% of cells, which is what the earlier branch's item 3 hit.
//      The contraction costs nothing to give up here: the kernel moves ~590 MB forward and is
//      bandwidth-bound, so the extra ALU instructions hide entirely.
//   2. The magnitude is `hypotf`, NOT `sqrtf(re*re + im*im)`. PyTorch's complex `abs` is hypot
//      (measured: identical to torch.hypot, while the naive form differs at 8.7e-8 on 11% of
//      cells). The two are not interchangeable at this tolerance.
//
// The backward reproduces autograd's ORDER, not its algebra; the algebraically collapsed form
// is a different rounding. One block per (unit, plane, row), striding over angle x elevation,
// so the floor and the rms are read once per row and no per-cell integer division is needed.
// `grad_s` is returned per cell rather than reduced here, so the floor's gradient keeps
// PyTorch's own reduction and stays bitwise.

__global__ void rae_plane_term_forward_kernel(
    const float2* __restrict__ spec,
    const float* __restrict__ floor_row, const __half* __restrict__ target,
    const float* __restrict__ rms, float* __restrict__ out,
    float factor, float eps, int ae, int planes_rows, int rows)
{
    const int row = blockIdx.x;                       // (unit, plane, row), row-major
    const int unit = row / planes_rows;
    const float f = __fmul_rn(floor_row[row], factor);
    const float r = rms[unit];
    const long long base = (long long)row * ae;
    for (int i = threadIdx.x; i < ae; i += blockDim.x) {
        const long long k = base + i;
        const float2 z = spec[k];
        const float mag = hypotf(z.x, z.y);
        const float s = __fadd_rn(__fmul_rn(mag, mag), f);
        const float pred = sqrtf(fmaxf(s, eps));
        const float t = __fmul_rn(__half2float(target[k]), r);
        // the DIFFERENCE, not its square: the caller squares it, so that the windowed term can
        // square AFTER masking and reproduce main's gradient chain exactly
        out[k] = __fsub_rn(pred, t);
    }
}

__global__ void rae_plane_term_backward_kernel(
    const float* __restrict__ grad_out, const float2* __restrict__ spec,
    const float* __restrict__ floor_row, const __half* __restrict__ target,
    const float* __restrict__ rms,
    float2* __restrict__ grad_spec, float* __restrict__ grad_s,
    float factor, float eps, int ae, int planes_rows, int rows)
{
    const int row = blockIdx.x;
    const int unit = row / planes_rows;
    const float f = __fmul_rn(floor_row[row], factor);
    const float r = rms[unit];
    const long long base = (long long)row * ae;
    for (int i = threadIdx.x; i < ae; i += blockDim.x) {
        const long long k = base + i;
        const float2 z = spec[k];
        const float a = z.x, b = z.y;
        const float mag = hypotf(a, b);
        const float s = __fadd_rn(__fmul_rn(mag, mag), f);
        const float pred = sqrtf(fmaxf(s, eps));
        // autograd's order from the difference down: sqrt -> clamp_min -> square -> abs. The
        // square that used to head this chain now lives outside the kernel, so `grad_out` is
        // already d(loss)/d(pred - target).
        const float g2 = __fdiv_rn(grad_out[k], __fmul_rn(2.0f, pred));
        const float g3 = (s > eps) ? g2 : 0.0f;
        const float g4 = __fmul_rn(__fmul_rn(g3, 2.0f), mag);
        // abs's backward is grad * sgn(z), and PyTorch defines the complex sgn of zero as ZERO.
        // An out-of-band plane's cell really is exactly 0 on real data, so dividing by the
        // magnitude unguarded makes 0/0 = NaN -- which is what the first build did, and the
        // random-input test never saw because randn does not produce exact zeros.
        const float sr = (mag != 0.0f) ? __fdiv_rn(a, mag) : 0.0f;
        const float si = (mag != 0.0f) ? __fdiv_rn(b, mag) : 0.0f;
        grad_spec[k] = make_float2(__fmul_rn(g4, sr), __fmul_rn(g4, si));
        grad_s[k] = g3;
    }
}

void launch_rae_plane_term_forward(
    const void* spec, const float* floor_row,
    const void* target, const float* rms, float* out,
    float factor, float eps, int ae, int planes_rows, int rows, cudaStream_t stream)
{
    const int block = (ae >= 256) ? 256 : 64;
    rae_plane_term_forward_kernel<<<(unsigned)rows, block, 0, stream>>>(
        reinterpret_cast<const float2*>(spec), floor_row,
        reinterpret_cast<const __half*>(target), rms, out,
        factor, eps, ae, planes_rows, rows);
}

void launch_rae_plane_term_backward(
    const float* grad_out, const void* spec, const float* floor_row,
    const void* target, const float* rms,
    void* grad_spec, float* grad_s,
    float factor, float eps, int ae, int planes_rows, int rows, cudaStream_t stream)
{
    const int block = (ae >= 256) ? 256 : 64;
    rae_plane_term_backward_kernel<<<(unsigned)rows, block, 0, stream>>>(
        grad_out, reinterpret_cast<const float2*>(spec), floor_row,
        reinterpret_cast<const __half*>(target), rms,
        reinterpret_cast<float2*>(grad_spec), grad_s, factor, eps, ae, planes_rows, rows);
}

// ---------------------------------------------------------------------------
// The RAE view's 2D-FFT input, built directly (docs/RUNTIME_STEP_20MS_PLAN.md item C).
//
// `_rae_spectrum_input_eager` gathers through the (n_az * n_el_slots, width) slot table -- 602 x 2
// on the cascade -- and materialises a (units, planes, 602, 2, rows) intermediate of 565 MiB a
// step whose slots are 78% EMPTY: 192 channels occupy 134 of 602 slots, and the intermediate is
// 3.5x the 161 MiB buffer the transform actually consumes. Measured, that kernel costs 8.79 ms a
// step, about a tenth of achievable bandwidth.
//
// Here each output cell of the (n_angle, n_el) grid reads its own one or two channels directly:
// the channel -> slot map is a PARTITION (cascade: 76 slots hold one channel, 58 hold two), so no
// cell is written twice and nothing accumulates. One block per (lead, row) stages nothing and
// writes 688 complex cells; the strided channel reads are absorbed by L2, since the 244 blocks of
// one (unit, plane) share a 187 KiB slab.
//
// BITWISE the table form, by construction and by MM4DPS_RAE_CHECK: multiplying a padded member by
// 0.0 and adding it is an exact no-op, so dropping the mask changes nothing, and the surviving
// arithmetic is the same sum and the same weight multiply in the same order. Round-to-nearest
// intrinsics keep nvcc from contracting the sum and the multiply into one rounding.

__global__ void rae_spectrum_input_kernel(
    const float2* __restrict__ spec, const long* __restrict__ range_index,
    const int* __restrict__ ch_a, const int* __restrict__ ch_b,
    const float* __restrict__ w_slot, float2* __restrict__ out,
    int rows, int ae, int channels, int bins, int lead_per_unit)
{
    const int lr = blockIdx.x;                 // (flat lead index, row)
    const int lead = lr / rows;
    const int row = lr - lead * rows;
    const long k = range_index[(long)(lead / lead_per_unit) * rows + row];
    // the profile is read as complex, one 8-byte load a channel: passing .real/.imag instead
    // makes the caller materialise two contiguous copies of a stride-2 view, measured at
    // 1.26 ms a call (docs/RUNTIME_STEP_20MS_PLAN.md 12)
    const float2* z = spec + (long)lead * channels * bins;
    float2* o = out + (long)lr * ae;
    for (int cell = threadIdx.x; cell < ae; cell += blockDim.x) {
        const int a = ch_a[cell], b = ch_b[cell];
        float accr = 0.0f, acci = 0.0f;
        if (a >= 0) { const float2 v = z[(long)a * bins + k]; accr = v.x; acci = v.y; }
        if (b >= 0) {
            const float2 v = z[(long)b * bins + k];
            accr = __fadd_rn(accr, v.x);
            acci = __fadd_rn(acci, v.y);
        }
        const float w = w_slot[cell];
        o[cell] = make_float2(__fmul_rn(accr, w), __fmul_rn(acci, w));
    }
}

// The backward: each channel belongs to one cell, so this is a permutation times the slot weight.
// The gradient buffers are zeroed by the caller, because only the kept range bins are written.
__global__ void rae_spectrum_input_backward_kernel(
    const float2* __restrict__ grad_out, const long* __restrict__ range_index,
    const int* __restrict__ cell_of_channel, const float* __restrict__ w_slot,
    float2* __restrict__ grad_spec,
    int rows, int ae, int channels, int bins, int lead_per_unit)
{
    const int lr = blockIdx.x;
    const int lead = lr / rows;
    const int row = lr - lead * rows;
    const long k = range_index[(long)(lead / lead_per_unit) * rows + row];
    const float2* g = grad_out + (long)lr * ae;
    float2* gz = grad_spec + (long)lead * channels * bins;
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        const int cell = cell_of_channel[c];
        if (cell < 0) continue;
        const float w = w_slot[cell];
        const float2 gc = g[cell];
        gz[(long)c * bins + k] = make_float2(__fmul_rn(gc.x, w), __fmul_rn(gc.y, w));
    }
}

void launch_rae_spectrum_input(
    const void* spec, const long* range_index,
    const int* ch_a, const int* ch_b, const float* w_slot, void* out,
    int lead, int rows, int ae, int channels, int bins, int lead_per_unit, cudaStream_t stream)
{
    const int block = (ae >= 256) ? 256 : 64;
    rae_spectrum_input_kernel<<<(unsigned)(lead * rows), block, 0, stream>>>(
        reinterpret_cast<const float2*>(spec), range_index, ch_a, ch_b, w_slot,
        reinterpret_cast<float2*>(out), rows, ae, channels, bins, lead_per_unit);
}

void launch_rae_spectrum_input_backward(
    const void* grad_out, const long* range_index, const int* cell_of_channel,
    const float* w_slot, void* grad_spec,
    int lead, int rows, int ae, int channels, int bins, int lead_per_unit, cudaStream_t stream)
{
    const int block = (channels >= 256) ? 256 : 64;
    rae_spectrum_input_backward_kernel<<<(unsigned)(lead * rows), block, 0, stream>>>(
        reinterpret_cast<const float2*>(grad_out), range_index, cell_of_channel, w_slot,
        reinterpret_cast<float2*>(grad_spec), rows, ae, channels, bins, lead_per_unit);
}

} // namespace mm25v5
