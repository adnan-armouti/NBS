// PyBind entry points for the HuPR CUDA kernels.
//
// These nine wrappers and their launcher declarations lived in
// third_party/3DPS_submission/mm3DPS/cuda/bindings.cpp, which made a 1,006-line
// file out of upstream's 542. They are ours; they call only mm25v5:: launchers
// defined in kernels.cu, and nothing upstream. They build the `hupr_cuda`
// extension, separate from `mm3DPS_cuda`.
//
// The CHECK_* macros below are the standard torch-extension idiom written out
// here, not copied from upstream -- one TORCH_CHECK each.
#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

namespace mm25v5 {



void launch_static_gain_splat_forward(
    const float* contrib_real, const float* contrib_imag,
    const int* flat_indices, const int64_t* point_indices,
    const float* gains, float* out_real, float* out_imag,
    int64_t spread, int64_t paths, cudaStream_t stream);

void launch_static_gain_splat_backward(
    const float* grad_out_real, const float* grad_out_imag,
    const float* contrib_real, const float* contrib_imag,
    const int* flat_indices, const int64_t* point_indices,
    const float* gains, float* grad_gains,
    float* grad_contrib_real, float* grad_contrib_imag,
    int64_t spread, int64_t paths, cudaStream_t stream);

void launch_static_scene_frames_forward(
    const int* runs, const int* entry, const float* carrier,
    const float* table4, const float* gains_t,
    float* out, int64_t base, int K, int half, int n_grid, int frames,
    cudaStream_t stream);

void launch_static_scene_frames_backward(
    const float* grad_out_t, const int* entry, const float* table4,
    const float* gains_t, float* grad_carrier,
    int64_t paths, int K, int half, int n_grid, int frames, cudaStream_t stream);

void launch_doppler_plane_sorted_forward(
    const int* runs, float* record, const int* path_of_entry,
    const int64_t* tx_indices, const float* path_rates, float phase_const,
    float* amps_e, const int64_t* unit_of_path,
    const int* active_planes, const int* bin_to_plane, int windowed, int anchor,
    const double* tx_times, int taps, const float* table4,
    float* out_real, float* out_imag,
    int64_t entries, int64_t cells, int64_t unit_base, int K, int spread,
    int n_grid, int planes, int L, cudaStream_t stream);

void launch_doppler_plane_sorted_backward(
    const float* grad_out_real, const float* grad_out_imag, float* grad_scratch,
    const int* path_of_entry, const float* record, const float* amps_e,
    const int64_t* tx_indices, const float* path_rates,
    const int64_t* unit_of_path, const int* active_planes, const int* bin_to_plane,
    const float* table4, float phase_const,
    float* grad_carrier, float* grad_rates,
    int64_t entries, int64_t cells, int64_t unit_base, int channels_per_point,
    int spread, int K, int n_grid, int planes, int L, int taps, int windowed,
    int anchor, const double* tx_times, cudaStream_t stream);

void launch_slow_time_splat_forward(
    const float* contrib_real, const float* contrib_imag,
    const int* flat_indices, const int64_t* tx_indices,
    const float* path_rates, const float* times, float phase_const,
    float* out_real, float* out_imag,
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base, cudaStream_t stream);

void launch_slow_time_splat_backward(
    const float* grad_out_real, const float* grad_out_imag,
    const float* contrib_real, const float* contrib_imag,
    const int* flat_indices, const int64_t* tx_indices,
    const float* path_rates, const float* times, float phase_const,
    float* grad_contrib_real, float* grad_contrib_imag, float* grad_rates,
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base, cudaStream_t stream);

void launch_doppler_plane_splat_forward(
    const float* contrib_real, const float* contrib_imag,
    const int* flat_indices, const int64_t* tx_indices,
    const float* path_rates, const float* times,
    const float* weight_real, const float* weight_imag, float phase_const,
    float* out_real, float* out_imag,
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base, int64_t planes, cudaStream_t stream);

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
    const int64_t* unit_of_path = nullptr,
    const int* active_planes = nullptr,
    const int* plane_bins = nullptr, int windowed = 0, int anchor = 0,
    const double* tx_times = nullptr,
    const int* bin_to_plane = nullptr, int taps = 0);

void launch_doppler_plane_gather_forward(
    const float* contrib_real, const float* contrib_imag,
    const int* flat_indices, const int64_t* tx_indices,
    const float* path_rates, const float* times,
    const float* weight_real, const float* weight_imag, float phase_const,
    float* amps, int* counts, int* offsets, int* cursor, float* slots,
    float* out_real, float* out_imag,
    int64_t spread, int64_t paths, int64_t loops, int64_t n_tx,
    int64_t base, int64_t planes, int64_t sources, int cursor_blocks,
    cudaStream_t stream, const int64_t* unit_of_path = nullptr,
    const int* active_planes = nullptr, int64_t unit_base = 0,
    const int* plane_bins = nullptr, int windowed = 0, int anchor = 0,
    const double* tx_times = nullptr,
    const int* bin_to_plane = nullptr, int taps = 0, int* n_of_path = nullptr);

void launch_rae_spectrum_input(
    const void* spec, const long* range_index,
    const int* ch_a, const int* ch_b, const float* w_slot, void* out,
    int lead, int rows, int ae, int channels, int bins, int lead_per_unit, cudaStream_t stream);

void launch_rae_spectrum_input_backward(
    const void* grad_out, const long* range_index, const int* cell_of_channel,
    const float* w_slot, void* grad_spec,
    int lead, int rows, int ae, int channels, int bins, int lead_per_unit, cudaStream_t stream);

void launch_rae_plane_term_forward(
    const void* spec, const float* floor_row,
    const void* target, const float* rms, float* out,
    float factor, float eps, int ae, int planes_rows, int rows, cudaStream_t stream);

void launch_rae_plane_term_backward(
    const float* grad_out, const void* spec, const float* floor_row,
    const void* target, const float* rms,
    void* grad_spec, float* grad_s,
    float factor, float eps, int ae, int planes_rows, int rows, cudaStream_t stream);

void launch_static_gain_map_forward(
    const int* cell_offsets, const float* slot_contrib, const int* slot_point,
    const float* gains, float* out_real, float* out_imag,
    int64_t base, cudaStream_t stream);

void launch_static_gain_map_backward(
    const float* grad_out_real, const float* grad_out_imag,
    const float* contrib_real, const float* contrib_imag,
    const int* flat_indices, const int* point_offsets,
    const int* point_paths, const float* gains, float* grad_gains,
    float* grad_contrib_real, float* grad_contrib_imag,
    int64_t spread, int64_t paths, int64_t n_points, cudaStream_t stream);
} // namespace mm25v5

#define CHECK_CUDA(x)    TORCH_CHECK(x.is_cuda(),       #x " must be a CUDA tensor")
#define CHECK_CONTIG(x)  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INT64(x)   TORCH_CHECK(x.scalar_type() == at::kLong,  #x " must be int64")
#define CHECK_INT32(x)   TORCH_CHECK(x.scalar_type() == at::kInt,   #x " must be int32")

// ---- HuPR K1: gain-weighted static scatter ----
std::vector<torch::Tensor> hupr_static_gain_splat_forward(
    torch::Tensor contrib_real, torch::Tensor contrib_imag,
    torch::Tensor flat_indices, torch::Tensor point_indices,
    torch::Tensor gains, int64_t base)
{
    CHECK_CUDA(contrib_real); CHECK_CONTIG(contrib_real); CHECK_FLOAT32(contrib_real);
    CHECK_CUDA(contrib_imag); CHECK_CONTIG(contrib_imag); CHECK_FLOAT32(contrib_imag);
    CHECK_CUDA(flat_indices); CHECK_CONTIG(flat_indices); CHECK_INT32(flat_indices);
    CHECK_CUDA(point_indices); CHECK_CONTIG(point_indices); CHECK_INT64(point_indices);
    CHECK_CUDA(gains); CHECK_CONTIG(gains); CHECK_FLOAT32(gains);
    const int64_t spread = contrib_real.size(0);
    const int64_t paths = contrib_real.size(1);
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(contrib_real.device());
    auto out_real = torch::zeros({base}, opt);
    auto out_imag = torch::zeros({base}, opt);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_static_gain_splat_forward(
        contrib_real.data_ptr<float>(), contrib_imag.data_ptr<float>(),
        flat_indices.data_ptr<int>(), point_indices.data_ptr<int64_t>(),
        gains.data_ptr<float>(), out_real.data_ptr<float>(), out_imag.data_ptr<float>(),
        spread, paths, stream.stream());
    return {out_real, out_imag};
}

std::vector<torch::Tensor> hupr_static_gain_splat_backward(
    torch::Tensor grad_out_real, torch::Tensor grad_out_imag,
    torch::Tensor contrib_real, torch::Tensor contrib_imag,
    torch::Tensor flat_indices, torch::Tensor point_indices,
    torch::Tensor gains, bool need_gains)
{
    const int64_t spread = contrib_real.size(0);
    const int64_t paths = contrib_real.size(1);
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(contrib_real.device());
    auto grad_gains = need_gains ? torch::zeros_like(gains)
                                 : torch::empty({0}, opt);
    auto grad_c_real = torch::empty({spread, paths}, opt);
    auto grad_c_imag = torch::empty({spread, paths}, opt);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_static_gain_splat_backward(
        grad_out_real.contiguous().data_ptr<float>(),
        grad_out_imag.contiguous().data_ptr<float>(),
        contrib_real.data_ptr<float>(), contrib_imag.data_ptr<float>(),
        flat_indices.data_ptr<int>(), point_indices.data_ptr<int64_t>(),
        gains.data_ptr<float>(),
        need_gains ? grad_gains.data_ptr<float>() : nullptr,
        grad_c_real.data_ptr<float>(), grad_c_imag.data_ptr<float>(),
        spread, paths, stream.stream());
    return {grad_gains, grad_c_real, grad_c_imag};
}

// ---- HuPR K2: fused slow-time velocity splat ----
std::vector<torch::Tensor> hupr_slow_time_splat_forward(
    torch::Tensor contrib_real, torch::Tensor contrib_imag,
    torch::Tensor flat_indices, torch::Tensor tx_indices,
    torch::Tensor path_rates, torch::Tensor times,
    double phase_const, int64_t base)
{
    CHECK_CUDA(contrib_real); CHECK_CONTIG(contrib_real); CHECK_FLOAT32(contrib_real);
    CHECK_CUDA(path_rates); CHECK_CONTIG(path_rates); CHECK_FLOAT32(path_rates);
    CHECK_CUDA(times); CHECK_CONTIG(times); CHECK_FLOAT32(times);
    CHECK_CUDA(flat_indices); CHECK_CONTIG(flat_indices); CHECK_INT32(flat_indices);
    CHECK_CUDA(tx_indices); CHECK_CONTIG(tx_indices); CHECK_INT64(tx_indices);
    const int64_t spread = contrib_real.size(0);
    const int64_t paths = contrib_real.size(1);
    const int64_t loops = times.size(0);
    const int64_t n_tx = times.size(1);
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(contrib_real.device());
    auto out_real = torch::zeros({loops * base}, opt);
    auto out_imag = torch::zeros({loops * base}, opt);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_slow_time_splat_forward(
        contrib_real.data_ptr<float>(), contrib_imag.data_ptr<float>(),
        flat_indices.data_ptr<int>(), tx_indices.data_ptr<int64_t>(),
        path_rates.data_ptr<float>(), times.data_ptr<float>(),
        (float)phase_const,
        out_real.data_ptr<float>(), out_imag.data_ptr<float>(),
        spread, paths, loops, n_tx, base, stream.stream());
    return {out_real, out_imag};
}

std::vector<torch::Tensor> hupr_slow_time_splat_backward(
    torch::Tensor grad_out_real, torch::Tensor grad_out_imag,
    torch::Tensor contrib_real, torch::Tensor contrib_imag,
    torch::Tensor flat_indices, torch::Tensor tx_indices,
    torch::Tensor path_rates, torch::Tensor times,
    double phase_const, int64_t base)
{
    const int64_t spread = contrib_real.size(0);
    const int64_t paths = contrib_real.size(1);
    const int64_t loops = times.size(0);
    const int64_t n_tx = times.size(1);
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(contrib_real.device());
    auto grad_c_real = torch::empty({spread, paths}, opt);
    auto grad_c_imag = torch::empty({spread, paths}, opt);
    auto grad_rates = torch::empty({paths}, opt);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_slow_time_splat_backward(
        grad_out_real.contiguous().data_ptr<float>(),
        grad_out_imag.contiguous().data_ptr<float>(),
        contrib_real.data_ptr<float>(), contrib_imag.data_ptr<float>(),
        flat_indices.data_ptr<int>(), tx_indices.data_ptr<int64_t>(),
        path_rates.data_ptr<float>(), times.data_ptr<float>(),
        (float)phase_const,
        grad_c_real.data_ptr<float>(), grad_c_imag.data_ptr<float>(),
        grad_rates.data_ptr<float>(),
        spread, paths, loops, n_tx, base, stream.stream());
    return {grad_c_real, grad_c_imag, grad_rates};
}

// ---- HuPR K2b (E1): splat into the requested Doppler bins only ----
// E9b (Phase 5 W2): K1 through the fixed sparse map. `cell_offsets` /
// `slot_contrib` / `slot_point` are the scene's static scatter, sorted into
// destination order once at startup; only `gains` varies.
// ---- the RAE view's 2D-FFT input, built straight from the channels (bitwise) ----
// `real`/`imag` are (lead..., channels, bins); `range_index` is (units, rows). Returns the complex
// (lead..., rows, n_angle, n_el) buffer the two FFTs consume.
torch::Tensor hupr_rae_spectrum_input(
    torch::Tensor spec, torch::Tensor range_index,
    torch::Tensor ch_a, torch::Tensor ch_b, torch::Tensor w_slot,
    int64_t n_angle, int64_t n_el, int64_t lead_per_unit)
{
    CHECK_CUDA(spec); CHECK_CONTIG(spec);
    TORCH_CHECK(spec.scalar_type() == at::kComplexFloat, "spec must be complex64");
    CHECK_CUDA(range_index); CHECK_CONTIG(range_index); CHECK_INT64(range_index);
    CHECK_CUDA(ch_a); CHECK_CONTIG(ch_a); CHECK_INT32(ch_a);
    CHECK_CUDA(ch_b); CHECK_CONTIG(ch_b); CHECK_INT32(ch_b);
    CHECK_CUDA(w_slot); CHECK_CONTIG(w_slot); CHECK_FLOAT32(w_slot);
    const auto channels = spec.size(-2), bins = spec.size(-1);
    const auto rows = range_index.size(-1);
    int64_t lead = 1;
    for (int64_t d = 0; d < spec.dim() - 2; ++d) lead *= spec.size(d);
    TORCH_CHECK(ch_a.numel() == n_angle * n_el && ch_b.numel() == n_angle * n_el
                && w_slot.numel() == n_angle * n_el, "the slot tables must cover n_angle x n_el");
    std::vector<int64_t> shape;
    for (int64_t d = 0; d < spec.dim() - 2; ++d) shape.push_back(spec.size(d));
    shape.push_back(rows); shape.push_back(n_angle); shape.push_back(n_el);
    auto out = torch::empty(shape, torch::TensorOptions()
                            .dtype(torch::kComplexFloat).device(spec.device()));
    mm25v5::launch_rae_spectrum_input(
        spec.data_ptr(), range_index.data_ptr<int64_t>(),
        ch_a.data_ptr<int>(), ch_b.data_ptr<int>(), w_slot.data_ptr<float>(), out.data_ptr(),
        (int)lead, (int)rows, (int)(n_angle * n_el), (int)channels, (int)bins, (int)lead_per_unit,
        at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor hupr_rae_spectrum_input_backward(
    torch::Tensor grad_out, torch::Tensor range_index, torch::Tensor cell_of_channel,
    torch::Tensor w_slot, int64_t channels, int64_t bins, int64_t lead_per_unit)
{
    CHECK_CUDA(grad_out); CHECK_CONTIG(grad_out);
    TORCH_CHECK(grad_out.scalar_type() == at::kComplexFloat, "grad_out must be complex64");
    const auto rows = range_index.size(-1);
    const auto n_angle = grad_out.size(-2), n_el = grad_out.size(-1);
    int64_t lead = 1;
    for (int64_t d = 0; d < grad_out.dim() - 3; ++d) lead *= grad_out.size(d);
    std::vector<int64_t> shape;
    for (int64_t d = 0; d < grad_out.dim() - 3; ++d) shape.push_back(grad_out.size(d));
    shape.push_back(channels); shape.push_back(bins);
    // zeroed, because only the kept range bins are written
    auto grad_spec = torch::zeros(shape, torch::TensorOptions()
                                  .dtype(torch::kComplexFloat).device(grad_out.device()));
    mm25v5::launch_rae_spectrum_input_backward(
        grad_out.data_ptr(), range_index.data_ptr<int64_t>(), cell_of_channel.data_ptr<int>(),
        w_slot.data_ptr<float>(), grad_spec.data_ptr(),
        (int)lead, (int)rows, (int)(n_angle * n_el), (int)channels, (int)bins, (int)lead_per_unit,
        at::cuda::getCurrentCUDAStream());
    return grad_spec;
}

// ---- the elevation term's plane part: the pointwise chain in one pass (bitwise) ----
// `spectrum` is the RAE view as a complex tensor, (units, planes, rows, angle, elevation);
// `floor_row` is its learned noise floor per (unit, plane, row); `target` the half-precision
// store and `rms` its per-unit denormaliser. Returns the per-cell squared difference, which the
// caller reduces exactly as before.
torch::Tensor hupr_rae_plane_term_forward(
    torch::Tensor spectrum, torch::Tensor floor_row, torch::Tensor target,
    torch::Tensor rms, double factor, double eps)
{
    CHECK_CUDA(spectrum); CHECK_CONTIG(spectrum);
    TORCH_CHECK(spectrum.scalar_type() == at::kComplexFloat, "spectrum must be complex64");
    TORCH_CHECK(spectrum.dim() == 5, "spectrum must be (units, planes, rows, angle, elevation)");
    CHECK_CUDA(floor_row); CHECK_CONTIG(floor_row); CHECK_FLOAT32(floor_row);
    CHECK_CUDA(target); CHECK_CONTIG(target);
    TORCH_CHECK(target.scalar_type() == at::kHalf, "target must be float16");
    CHECK_CUDA(rms); CHECK_CONTIG(rms); CHECK_FLOAT32(rms);
    const auto U = spectrum.size(0), P = spectrum.size(1), R = spectrum.size(2);
    const auto A = spectrum.size(3), E = spectrum.size(4);
    TORCH_CHECK(floor_row.numel() == U * P * R, "floor_row must hold one value per (unit, plane, row)");
    TORCH_CHECK(target.sizes() == spectrum.sizes(), "target must match the spectrum's shape");
    TORCH_CHECK(rms.numel() == U, "rms must hold one value per unit");
    auto out = torch::empty_like(target, target.options().dtype(torch::kFloat32));
    mm25v5::launch_rae_plane_term_forward(
        spectrum.data_ptr(), floor_row.data_ptr<float>(),
        target.data_ptr(), rms.data_ptr<float>(), out.data_ptr<float>(),
        (float)factor, (float)eps, (int)(A * E), (int)(P * R), (int)(U * P * R),
        at::cuda::getCurrentCUDAStream());
    return out;
}

std::vector<torch::Tensor> hupr_rae_plane_term_backward(
    torch::Tensor grad_out, torch::Tensor spectrum, torch::Tensor floor_row,
    torch::Tensor target, torch::Tensor rms, double factor, double eps)
{
    CHECK_CUDA(grad_out); CHECK_CONTIG(grad_out); CHECK_FLOAT32(grad_out);
    CHECK_CUDA(spectrum); CHECK_CONTIG(spectrum);
    const auto U = spectrum.size(0), P = spectrum.size(1), R = spectrum.size(2);
    const auto A = spectrum.size(3), E = spectrum.size(4);
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(spectrum.device());
    auto grad_spec = torch::empty_like(spectrum);
    auto grad_s = torch::empty({U, P, R, A, E}, opt);
    mm25v5::launch_rae_plane_term_backward(
        grad_out.data_ptr<float>(), spectrum.data_ptr(),
        floor_row.data_ptr<float>(), target.data_ptr(), rms.data_ptr<float>(),
        grad_spec.data_ptr(), grad_s.data_ptr<float>(),
        (float)factor, (float)eps, (int)(A * E), (int)(P * R), (int)(U * P * R),
        at::cuda::getCurrentCUDAStream());
    return {grad_spec, grad_s};
}

std::vector<torch::Tensor> hupr_static_gain_map_forward(
    torch::Tensor cell_offsets, torch::Tensor slot_contrib,
    torch::Tensor slot_point, torch::Tensor gains, int64_t base)
{
    CHECK_CUDA(slot_contrib); CHECK_CONTIG(slot_contrib); CHECK_FLOAT32(slot_contrib);
    CHECK_CUDA(gains); CHECK_CONTIG(gains); CHECK_FLOAT32(gains);
    CHECK_CUDA(cell_offsets); CHECK_CONTIG(cell_offsets);
    CHECK_CUDA(slot_point); CHECK_CONTIG(slot_point);
    TORCH_CHECK(cell_offsets.numel() == base + 1, "cell_offsets must be (base + 1)");
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(gains.device());
    auto out_real = torch::empty({base}, opt);
    auto out_imag = torch::empty({base}, opt);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_static_gain_map_forward(
        cell_offsets.data_ptr<int>(), slot_contrib.data_ptr<float>(),
        slot_point.data_ptr<int>(), gains.data_ptr<float>(),
        out_real.data_ptr<float>(), out_imag.data_ptr<float>(),
        base, stream.stream());
    return {out_real, out_imag};
}

torch::Tensor hupr_static_scene_frames_forward(
    torch::Tensor runs, torch::Tensor entry, torch::Tensor carrier,
    torch::Tensor table4, torch::Tensor gains_t,
    int64_t base, int64_t K)
{
    CHECK_CUDA(runs); CHECK_CONTIG(runs); CHECK_INT32(runs);
    CHECK_CUDA(entry); CHECK_CONTIG(entry); CHECK_INT32(entry);
    CHECK_CUDA(carrier); CHECK_CONTIG(carrier); CHECK_FLOAT32(carrier);
    CHECK_CUDA(table4); CHECK_CONTIG(table4); CHECK_FLOAT32(table4);
    TORCH_CHECK(table4.dim() == 3 && table4.size(2) == 4, "the PSF table must be (spread, n_grid, 4)");
    CHECK_CUDA(gains_t); CHECK_CONTIG(gains_t); CHECK_FLOAT32(gains_t);
    TORCH_CHECK(runs.dim() == 2 && runs.size(0) == base && runs.size(1) == 4, "runs must be (base, 4)");
    TORCH_CHECK(entry.dim() == 2 && entry.size(1) == 4, "entry must be (paths, 4)");
    TORCH_CHECK(carrier.dim() == 2 && carrier.size(1) == 2 && carrier.size(0) == entry.size(0),
                "carrier must be (paths, 2)");
    TORCH_CHECK(gains_t.dim() == 2 && gains_t.size(0) == entry.size(0)
                && gains_t.size(1) >= 1 && gains_t.size(1) <= 8,
                "gains must be (paths, frames <= 8), in entry order");
    TORCH_CHECK(table4.size(0) % 2 == 1, "the PSF table must have an odd number of taps");
    const int64_t frames = gains_t.size(1);
    auto out = torch::empty({frames, base, 2}, carrier.options());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_static_scene_frames_forward(
        runs.data_ptr<int>(), entry.data_ptr<int>(), carrier.data_ptr<float>(),
        table4.data_ptr<float>(),
        gains_t.data_ptr<float>(), out.data_ptr<float>(), base, (int)K,
        (int)(table4.size(0) / 2), (int)table4.size(1), (int)frames,
        stream.stream());
    return out;
}

torch::Tensor hupr_static_scene_frames_backward(
    torch::Tensor grad_out_t, torch::Tensor entry, torch::Tensor table4, torch::Tensor gains_t, int64_t K)
{
    CHECK_CUDA(grad_out_t); CHECK_CONTIG(grad_out_t); CHECK_FLOAT32(grad_out_t);
    CHECK_CUDA(entry); CHECK_CONTIG(entry); CHECK_INT32(entry);
    CHECK_CUDA(table4); CHECK_CONTIG(table4); CHECK_FLOAT32(table4);
    TORCH_CHECK(table4.dim() == 3 && table4.size(2) == 4, "the PSF table must be (spread, n_grid, 4)");
    CHECK_CUDA(gains_t); CHECK_CONTIG(gains_t); CHECK_FLOAT32(gains_t);
    TORCH_CHECK(grad_out_t.dim() == 3 && grad_out_t.size(2) == 2, "grad_out_t must be (base, frames, 2)");
    TORCH_CHECK(entry.dim() == 2 && entry.size(1) == 4, "entry must be (paths, 4)");
    const int64_t frames = grad_out_t.size(1);
    TORCH_CHECK(gains_t.size(1) == frames && frames <= 8, "gains_t must be (n_points, frames <= 8)");
    const int64_t paths = entry.size(0);
    auto grad_carrier = torch::empty({paths, 2}, grad_out_t.options());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_static_scene_frames_backward(
        grad_out_t.data_ptr<float>(), entry.data_ptr<int>(),
        table4.data_ptr<float>(),
        gains_t.data_ptr<float>(), grad_carrier.data_ptr<float>(), paths, (int)K,
        (int)(table4.size(0) / 2), (int)table4.size(1), (int)frames,
        stream.stream());
    return grad_carrier;
}

// The sorted Doppler route. `runs` (cells, 4) int32, `record` (entries, 4)
// float32, `path_of_entry` (entries,) int32; the rest as the gather forward.
// Returns {out_real, out_imag, amps_e, n_of_entry}; the backward takes the
// last two back.
std::vector<torch::Tensor> hupr_doppler_plane_sorted_forward(
    torch::Tensor runs, torch::Tensor record, torch::Tensor path_of_entry,
    torch::Tensor tx_indices, torch::Tensor path_rates, double phase_const,
    torch::Tensor unit_of_path, torch::Tensor active_planes,
    torch::Tensor bin_to_plane, int64_t windowed, int64_t anchor,
    torch::Tensor tx_times, int64_t taps, torch::Tensor table4, int64_t planes, int64_t unit_base, int64_t K,
    int64_t spread, int64_t loops)
{
    CHECK_CUDA(runs); CHECK_CONTIG(runs); CHECK_INT32(runs);
    CHECK_CUDA(record); CHECK_CONTIG(record); CHECK_FLOAT32(record);
    CHECK_CUDA(path_of_entry); CHECK_CONTIG(path_of_entry); CHECK_INT32(path_of_entry);
    CHECK_CUDA(tx_indices); CHECK_CONTIG(tx_indices);
    CHECK_CUDA(path_rates); CHECK_CONTIG(path_rates); CHECK_FLOAT32(path_rates);
    CHECK_CUDA(unit_of_path); CHECK_CONTIG(unit_of_path);
    CHECK_CUDA(active_planes); CHECK_CONTIG(active_planes); CHECK_INT32(active_planes);
    CHECK_CUDA(bin_to_plane); CHECK_CONTIG(bin_to_plane); CHECK_INT32(bin_to_plane);
    CHECK_CUDA(tx_times); CHECK_CONTIG(tx_times);
    CHECK_CUDA(table4); CHECK_CONTIG(table4); CHECK_FLOAT32(table4);
    TORCH_CHECK(table4.dim() == 3 && table4.size(2) == 4, "the PSF table must be (spread, n_grid, 4)");
    TORCH_CHECK(runs.dim() == 2 && runs.size(1) == 4, "runs must be (cells / 32, 4)");
    TORCH_CHECK(K % 32 == 0, "the sorted route needs a multiple of 32 range bins");
    TORCH_CHECK(record.dim() == 2 && record.size(1) == 4, "record must be (entries, 4)");
    TORCH_CHECK(record.size(0) == path_of_entry.numel(), "record and path_of_entry disagree");
    TORCH_CHECK(taps > 0 && taps <= 15 && (taps % 2) == 1, "taps must be odd, 1..15");
    TORCH_CHECK(planes <= 64, "the sorted route holds at most 64 planes");
    TORCH_CHECK(table4.size(0) == spread, "the PSF table must be (spread, n_grid)");
    TORCH_CHECK(tx_times.scalar_type() == at::kDouble, "tx_times must be float64");
    const int64_t entries = record.size(0);
    const int64_t cells = runs.size(0) * 32;
    const int64_t W = 3 + taps;
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(record.device());
    auto opt_i = torch::TensorOptions().dtype(torch::kInt32).device(record.device());
    auto out_real = torch::empty({planes * cells}, opt);
    auto out_imag = torch::empty({planes * cells}, opt);
    auto amps_e = torch::empty({entries * W * 2}, opt);
    (void)opt_i;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    // `record` is written in place: its fourth field takes the path's
    // integer Doppler bin beside the range bin (the caller builds it fresh
    // every call)
    mm25v5::launch_doppler_plane_sorted_forward(
        runs.data_ptr<int>(), record.data_ptr<float>(), path_of_entry.data_ptr<int>(),
        tx_indices.data_ptr<int64_t>(), path_rates.data_ptr<float>(), (float)phase_const,
        amps_e.data_ptr<float>(),
        unit_of_path.data_ptr<int64_t>(), active_planes.data_ptr<int>(),
        bin_to_plane.data_ptr<int>(), (int)windowed, (int)anchor,
        tx_times.data_ptr<double>(), (int)taps, table4.data_ptr<float>(), out_real.data_ptr<float>(), out_imag.data_ptr<float>(),
        entries, cells, unit_base, (int)K, (int)spread, (int)table4.size(1),
        (int)planes, (int)loops, stream.stream());
    return {out_real, out_imag, amps_e};
}

std::vector<torch::Tensor> hupr_doppler_plane_sorted_backward(
    torch::Tensor grad_out_real, torch::Tensor grad_out_imag,
    torch::Tensor path_of_entry, torch::Tensor record, torch::Tensor amps_e,
    torch::Tensor tx_indices, torch::Tensor path_rates, torch::Tensor unit_of_path,
    torch::Tensor active_planes, torch::Tensor bin_to_plane,
    torch::Tensor table4, double phase_const,
    int64_t planes, int64_t unit_base, int64_t channels_per_point, int64_t K,
    int64_t spread, int64_t loops, int64_t taps, int64_t windowed, int64_t anchor,
    torch::Tensor tx_times)
{
    CHECK_CUDA(grad_out_real); CHECK_FLOAT32(grad_out_real);
    CHECK_CUDA(record); CHECK_CONTIG(record); CHECK_FLOAT32(record);
    CHECK_CUDA(amps_e); CHECK_CONTIG(amps_e); CHECK_FLOAT32(amps_e);
    CHECK_CUDA(path_of_entry); CHECK_CONTIG(path_of_entry); CHECK_INT32(path_of_entry);
    const int64_t entries = record.size(0);
    const int64_t paths = path_rates.numel();
    const int64_t cells = grad_out_real.numel() / planes;
    TORCH_CHECK(amps_e.numel() == entries * (3 + taps) * 2, "amps_e must be (entries, 3 + taps) float2");
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(record.device());
    auto grad_carrier = torch::zeros({paths, 2}, opt);   // pads and unvisited paths stay zero
    auto grad_rates = torch::zeros({paths}, opt);
    auto scratch = torch::empty({cells * planes * 2}, opt);
    auto g_re = grad_out_real.contiguous();
    auto g_im = grad_out_imag.contiguous();
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_doppler_plane_sorted_backward(
        g_re.data_ptr<float>(), g_im.data_ptr<float>(), scratch.data_ptr<float>(),
        path_of_entry.data_ptr<int>(), record.data_ptr<float>(), amps_e.data_ptr<float>(),
        tx_indices.data_ptr<int64_t>(), path_rates.data_ptr<float>(),
        unit_of_path.data_ptr<int64_t>(), active_planes.data_ptr<int>(),
        bin_to_plane.data_ptr<int>(), table4.data_ptr<float>(), (float)phase_const,
        grad_carrier.data_ptr<float>(), grad_rates.data_ptr<float>(),
        entries, cells, unit_base, (int)channels_per_point, (int)spread, (int)K,
        (int)table4.size(1), (int)planes, (int)loops, (int)taps, (int)windowed,
        (int)anchor, tx_times.data_ptr<double>(), stream.stream());
    return {grad_carrier, grad_rates};
}

std::vector<torch::Tensor> hupr_static_gain_map_backward(
    torch::Tensor grad_out_real, torch::Tensor grad_out_imag,
    torch::Tensor contrib_real, torch::Tensor contrib_imag,
    torch::Tensor flat_indices, torch::Tensor point_offsets,
    torch::Tensor point_paths, torch::Tensor gains, bool need_contrib)
{
    CHECK_CUDA(grad_out_real); CHECK_CONTIG(grad_out_real); CHECK_FLOAT32(grad_out_real);
    CHECK_CUDA(contrib_real); CHECK_CONTIG(contrib_real); CHECK_FLOAT32(contrib_real);
    CHECK_CUDA(flat_indices); CHECK_CONTIG(flat_indices); CHECK_INT32(flat_indices);
    CHECK_CUDA(point_offsets); CHECK_CONTIG(point_offsets);
    CHECK_CUDA(point_paths); CHECK_CONTIG(point_paths);
    const int64_t spread = contrib_real.size(0);
    const int64_t paths = contrib_real.size(1);
    const int64_t n_points = gains.numel();
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(gains.device());
    auto grad_gains = torch::empty({n_points}, opt);
    // The scene's contributions are frozen, so the trainer never asks for
    // their gradient; computing it wrote 28 MB per launch for nothing.
    auto grad_contrib_real = need_contrib ? torch::empty({spread, paths}, opt)
                                          : torch::empty({0}, opt);
    auto grad_contrib_imag = need_contrib ? torch::empty({spread, paths}, opt)
                                          : torch::empty({0}, opt);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_static_gain_map_backward(
        grad_out_real.data_ptr<float>(), grad_out_imag.data_ptr<float>(),
        contrib_real.data_ptr<float>(), contrib_imag.data_ptr<float>(),
        flat_indices.data_ptr<int>(), point_offsets.data_ptr<int>(),
        point_paths.data_ptr<int>(), gains.data_ptr<float>(),
        grad_gains.data_ptr<float>(),
        need_contrib ? grad_contrib_real.data_ptr<float>() : nullptr,
        need_contrib ? grad_contrib_imag.data_ptr<float>() : nullptr,
        spread, paths, n_points, stream.stream());
    return {grad_gains, grad_contrib_real, grad_contrib_imag};
}

std::vector<torch::Tensor> hupr_doppler_plane_splat_forward(
    torch::Tensor contrib_real, torch::Tensor contrib_imag,
    torch::Tensor flat_indices, torch::Tensor tx_indices,
    torch::Tensor path_rates, torch::Tensor times,
    torch::Tensor weight_real, torch::Tensor weight_imag,
    double phase_const, int64_t base)
{
    CHECK_CUDA(contrib_real); CHECK_CONTIG(contrib_real); CHECK_FLOAT32(contrib_real);
    CHECK_CUDA(contrib_imag); CHECK_CONTIG(contrib_imag); CHECK_FLOAT32(contrib_imag);
    CHECK_CUDA(path_rates); CHECK_CONTIG(path_rates); CHECK_FLOAT32(path_rates);
    CHECK_CUDA(times); CHECK_CONTIG(times); CHECK_FLOAT32(times);
    CHECK_CUDA(weight_real); CHECK_CONTIG(weight_real); CHECK_FLOAT32(weight_real);
    CHECK_CUDA(weight_imag); CHECK_CONTIG(weight_imag); CHECK_FLOAT32(weight_imag);
    CHECK_CUDA(flat_indices); CHECK_CONTIG(flat_indices); CHECK_INT32(flat_indices);
    CHECK_CUDA(tx_indices); CHECK_CONTIG(tx_indices); CHECK_INT64(tx_indices);
    const int64_t spread = contrib_real.size(0);
    const int64_t paths = contrib_real.size(1);
    const int64_t loops = times.size(0);
    const int64_t n_tx = times.size(1);
    const int64_t planes = weight_real.size(-2);
    TORCH_CHECK(weight_real.size(-1) == loops,
                "weight matrix must be (planes, loops)");
    TORCH_CHECK(weight_imag.sizes() == weight_real.sizes(),
                "weight real/imag must have the same shape");
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(contrib_real.device());
    auto out_real = torch::zeros({planes * base}, opt);
    auto out_imag = torch::zeros({planes * base}, opt);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_doppler_plane_splat_forward(
        contrib_real.data_ptr<float>(), contrib_imag.data_ptr<float>(),
        flat_indices.data_ptr<int>(), tx_indices.data_ptr<int64_t>(),
        path_rates.data_ptr<float>(), times.data_ptr<float>(),
        weight_real.data_ptr<float>(), weight_imag.data_ptr<float>(),
        (float)phase_const,
        out_real.data_ptr<float>(), out_imag.data_ptr<float>(),
        spread, paths, loops, n_tx, base, planes, stream.stream());
    return {out_real, out_imag};
}

// E9a (Phase 5 W1): K2b's forward as a gather. Same arguments as the splat
// plus an optional leading `sources` axis on `path_rates` -- the evaluation
// pass's five velocity variants of a chunk share everything except the rates,
// so one call covers all five and the binning is paid for once.
//
// Returns (sources * planes * base,) real/imag, which the caller reshapes.
// Per-unit active plane counts: (units,) int32, only meaningful with a
// per-unit W. Returns the device pointer or nullptr, and checks that the
// units tile `base` exactly so a cell's unit is `cell / (base / units)`.
static const int* active_planes_ptr(
    const c10::optional<torch::Tensor>& active_planes, bool per_unit_w,
    int64_t planes, int64_t base, int64_t* unit_base)
{
    *unit_base = 0;
    if (!active_planes.has_value() || !active_planes->defined()
        || active_planes->numel() == 0)
        return nullptr;
    TORCH_CHECK(per_unit_w,
                "active_planes requires per-unit weights (unit_of_path)");
    const torch::Tensor& a = *active_planes;
    CHECK_CUDA(a); CHECK_CONTIG(a); CHECK_INT32(a);
    TORCH_CHECK(a.dim() == 1, "active_planes must be (units,)");
    TORCH_CHECK(base % a.numel() == 0,
                "active_planes: units must tile base");
    *unit_base = base / a.numel();
    return a.data_ptr<int>();
}

// Closed-form plane bins: (units, planes) int32, the fftshifted bin of every
// plane (-1 pads; entry 0 is the anchor and ignored). Non-null selects the
// closed-form kernels (doppler_closed_form.cuh); needs per-unit weights.
// The closed form's (n_tx, 2) float64 (t0, T) table.
static const double* tx_times_ptr(const c10::optional<torch::Tensor>& tx_times,
                                  const int* bins, int64_t n_tx)
{
    if (bins == nullptr) return nullptr;
    TORCH_CHECK(tx_times.has_value() && tx_times->defined() && tx_times->numel() == 2 * n_tx,
                "the closed form needs tx_times (n_tx, 2) float64");
    const torch::Tensor& t = *tx_times;
    CHECK_CUDA(t); CHECK_CONTIG(t);
    TORCH_CHECK(t.scalar_type() == at::kDouble, "tx_times must be float64");
    return t.data_ptr<double>();
}

// The window route's (units, loops + 1) int32 bin -> plane table.
static const int* bin_to_plane_ptr(const c10::optional<torch::Tensor>& table, bool per_unit_w,
                                   int64_t loops, const torch::Tensor& weight_real, int64_t taps)
{
    if (taps <= 0 || !table.has_value() || !table->defined() || table->numel() == 0)
        return nullptr;
    TORCH_CHECK(per_unit_w, "bin_to_plane (the window route) requires per-unit weights");
    TORCH_CHECK(taps % 2 == 1 && taps <= 15, "taps must be odd and at most 15");
    const torch::Tensor& t = *table;
    CHECK_CUDA(t); CHECK_CONTIG(t); CHECK_INT32(t);
    TORCH_CHECK(t.dim() == 2 && t.size(0) == weight_real.size(0) && t.size(1) == loops + 1,
                "bin_to_plane must be (units, loops + 1)");
    return t.data_ptr<int>();
}

static const int* plane_bins_ptr(
    const c10::optional<torch::Tensor>& plane_bins, bool per_unit_w,
    int64_t planes, const torch::Tensor& weight_real)
{
    if (!plane_bins.has_value() || !plane_bins->defined() || plane_bins->numel() == 0)
        return nullptr;
    TORCH_CHECK(per_unit_w, "plane_bins (the closed form) requires per-unit weights (unit_of_path)");
    const torch::Tensor& b = *plane_bins;
    CHECK_CUDA(b); CHECK_CONTIG(b); CHECK_INT32(b);
    TORCH_CHECK(b.dim() == 2 && b.size(0) == weight_real.size(0) && b.size(1) == planes,
                "plane_bins must be (units, planes)");
    return b.data_ptr<int>();
}

std::vector<torch::Tensor> hupr_doppler_plane_gather_forward(
    torch::Tensor contrib_real, torch::Tensor contrib_imag,
    torch::Tensor flat_indices, torch::Tensor tx_indices,
    torch::Tensor path_rates, torch::Tensor times,
    torch::Tensor weight_real, torch::Tensor weight_imag,
    double phase_const, int64_t base, int64_t deterministic,
    c10::optional<torch::Tensor> unit_of_path,
    c10::optional<torch::Tensor> active_planes,
    c10::optional<torch::Tensor> plane_bins, int64_t windowed, int64_t anchor,
    c10::optional<torch::Tensor> tx_times,
    c10::optional<torch::Tensor> bin_to_plane, int64_t taps)
{
    // Empty tensor = one W for every path (the historical call). Non-empty =
    // `weight_*` is (units, planes, loops) and this gives each path its unit.
    const bool per_unit_w = unit_of_path.has_value()
                         && unit_of_path->defined() && unit_of_path->numel() > 0;
    if (per_unit_w) {
        const torch::Tensor& unit_map = *unit_of_path;
        CHECK_CUDA(unit_map); CHECK_CONTIG(unit_map);
        CHECK_INT64(unit_map);
        TORCH_CHECK(weight_real.dim() == 3,
                    "per-unit weights must be (units, planes, loops)");
        if (active_planes.has_value() && active_planes->defined()
            && active_planes->numel() > 0)
            TORCH_CHECK(active_planes->numel() == weight_real.size(0),
                        "active_planes must have one entry per unit");
    }
    CHECK_CUDA(contrib_real); CHECK_CONTIG(contrib_real); CHECK_FLOAT32(contrib_real);
    CHECK_CUDA(contrib_imag); CHECK_CONTIG(contrib_imag); CHECK_FLOAT32(contrib_imag);
    CHECK_CUDA(path_rates); CHECK_CONTIG(path_rates); CHECK_FLOAT32(path_rates);
    CHECK_CUDA(times); CHECK_CONTIG(times); CHECK_FLOAT32(times);
    CHECK_CUDA(weight_real); CHECK_CONTIG(weight_real); CHECK_FLOAT32(weight_real);
    CHECK_CUDA(weight_imag); CHECK_CONTIG(weight_imag); CHECK_FLOAT32(weight_imag);
    CHECK_CUDA(flat_indices); CHECK_CONTIG(flat_indices); CHECK_INT32(flat_indices);
    CHECK_CUDA(tx_indices); CHECK_CONTIG(tx_indices); CHECK_INT64(tx_indices);
    const int64_t spread = contrib_real.size(0);
    const int64_t paths = contrib_real.size(1);
    const int64_t loops = times.size(0);
    const int64_t n_tx = times.size(1);
    const int64_t planes = weight_real.size(-2);
    const int64_t sources = (path_rates.dim() == 1) ? 1 : path_rates.size(0);
    // W4: with a deterministic binning `cursor` is a (blocks, base) histogram
    // instead of a (base,) arrival counter, so the block count trades
    // placement parallelism against histogram memory and against the scan
    // that walks it. Held to a 48 MB histogram and clamped: 341 blocks for a
    // training step's base of 36,864, 136 for an evaluation chunk's 92,160.
    const int cursor_blocks =
        deterministic ? (int)std::min<int64_t>(512,
            std::max<int64_t>(64, (48 << 20) / (base * (int64_t)sizeof(int))))
                      : 0;
    TORCH_CHECK(weight_real.size(-1) == loops,
                "weight matrix must be (planes, loops)");
    TORCH_CHECK(weight_imag.sizes() == weight_real.sizes(),
                "weight real/imag must have the same shape");
    TORCH_CHECK(path_rates.numel() == sources * paths,
                "path_rates must be (paths,) or (sources, paths)");
    TORCH_CHECK(sources * planes <= 1024,
                "sources * planes must fit one block (<= 1024)");
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(contrib_real.device());
    auto opt_i = torch::TensorOptions().dtype(torch::kInt32).device(contrib_real.device());
    auto out_real = torch::empty({sources * planes * base}, opt);
    auto out_imag = torch::empty({sources * planes * base}, opt);
    if (paths == 0) {
        out_real.zero_();
        out_imag.zero_();
        return {out_real, out_imag, torch::empty({0}, opt)};
    }
    const int* table = bin_to_plane_ptr(bin_to_plane, per_unit_w, loops, weight_real, taps);
    if (table != nullptr) {
        TORCH_CHECK(sources == 1, "the window route takes one velocity source");
        TORCH_CHECK(plane_bins.has_value() && plane_bins->defined() && plane_bins->numel() > 0,
                    "the window route is the closed form: plane_bins is required");
    }
    // Scratch. Under CUDA-graph capture these come from the graph's private
    // pool and their shapes are static, which is what capture requires.
    // The window route's amplitudes are (paths, 3 + taps), every entry
    // written, plus the path's integer Doppler bin, returned after the amps
    // (the backward reads both).
    auto amps = torch::empty({table != nullptr ? paths * (3 + taps) * 2
                                               : sources * paths * planes * 2}, opt);
    auto n_of_path = torch::empty({table != nullptr ? paths : 0}, opt_i);
    auto counts = torch::zeros({base}, opt_i);
    auto offsets = torch::empty({base + 1}, opt_i);
    auto cursor = torch::zeros({(cursor_blocks > 0 ? cursor_blocks : 1) * base},
                               opt_i);
    auto slots = torch::empty({spread * paths * 4}, opt);
    int64_t unit_base = 0;
    const int* active = active_planes_ptr(active_planes, per_unit_w, planes,
                                          base, &unit_base);
    const int* bins = plane_bins_ptr(plane_bins, per_unit_w, planes, weight_real);
    const double* tx_t = tx_times_ptr(tx_times, bins, n_tx);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_doppler_plane_gather_forward(
        contrib_real.data_ptr<float>(), contrib_imag.data_ptr<float>(),
        flat_indices.data_ptr<int>(), tx_indices.data_ptr<int64_t>(),
        path_rates.data_ptr<float>(), times.data_ptr<float>(),
        weight_real.data_ptr<float>(), weight_imag.data_ptr<float>(),
        (float)phase_const,
        amps.data_ptr<float>(), counts.data_ptr<int>(), offsets.data_ptr<int>(),
        cursor.data_ptr<int>(), slots.data_ptr<float>(),
        out_real.data_ptr<float>(), out_imag.data_ptr<float>(),
        spread, paths, loops, n_tx, base, planes, sources, cursor_blocks,
        stream.stream(),
        per_unit_w ? unit_of_path->data_ptr<int64_t>() : nullptr,
        active, unit_base, bins, (int)windowed, (int)anchor, tx_t,
        table, (int)taps, table != nullptr ? n_of_path.data_ptr<int>() : nullptr);
    // `amps` is returned so the backward can *read* the plane amplitudes the
    // forward already computed instead of rebuilding them per plane chunk;
    // the amplitudes kernel accumulates over `l` in the same order the
    // backward's rebuild loop did, so the values are bitwise identical.
    return {out_real, out_imag, amps, n_of_path};
}

std::vector<torch::Tensor> hupr_doppler_plane_splat_backward(
    torch::Tensor grad_out_real, torch::Tensor grad_out_imag,
    torch::Tensor contrib_real, torch::Tensor contrib_imag,
    torch::Tensor flat_indices, torch::Tensor tx_indices,
    torch::Tensor path_rates, torch::Tensor times,
    torch::Tensor weight_real, torch::Tensor weight_imag,
    double phase_const, int64_t base, int64_t tiled,
    torch::Tensor forward_amps,
    c10::optional<torch::Tensor> unit_of_path,
    c10::optional<torch::Tensor> active_planes,
    c10::optional<torch::Tensor> plane_bins, int64_t windowed, int64_t anchor,
    c10::optional<torch::Tensor> tx_times,
    c10::optional<torch::Tensor> bin_to_plane, int64_t taps)
{
    const bool per_unit_w = unit_of_path.has_value()
                         && unit_of_path->defined() && unit_of_path->numel() > 0;
    if (per_unit_w) {
        const torch::Tensor& unit_map = *unit_of_path;
        CHECK_CUDA(unit_map); CHECK_CONTIG(unit_map);
        CHECK_INT64(unit_map);
        TORCH_CHECK(weight_real.dim() == 3,
                    "per-unit weights must be (units, planes, loops)");
        // Per-unit W has exactly one kernel route -- the amps-fed chunked
        // backward -- so the conditions that route needs are requirements,
        // not preferences. Failing here is the point: a silent fall-through
        // to another variant would read the wrong frame's Doppler rows.
        TORCH_CHECK(forward_amps.defined() && forward_amps.numel() > 0,
                    "per-unit Doppler weights require the forward's saved "
                    "amplitudes; call the gather forward with the same map");
        TORCH_CHECK(contrib_real.size(0) <= 16,   // DOPPLER_CHUNK_MAX_SPREAD
                    "per-unit Doppler weights require spread <= 16, got ",
                    contrib_real.size(0));
    }
    const bool has_amps = forward_amps.defined() && forward_amps.numel() > 0;
    const bool window_route = taps > 0 && bin_to_plane.has_value() && bin_to_plane->defined()
                              && bin_to_plane->numel() > 0;
    if (has_amps) {
        CHECK_CUDA(forward_amps); CHECK_CONTIG(forward_amps);
        CHECK_FLOAT32(forward_amps);
        TORCH_CHECK(forward_amps.numel()
                        == contrib_real.size(1) * (window_route ? 3 + taps : weight_real.size(-2)) * 2,
                    "forward_amps must be (paths, planes) float2, or (paths, 3 + taps) on the window route");
    }
    TORCH_CHECK(!window_route || has_amps, "the window route needs the forward's amplitudes");
    const int64_t spread = contrib_real.size(0);
    const int64_t paths = contrib_real.size(1);
    const int64_t loops = times.size(0);
    const int64_t n_tx = times.size(1);
    const int64_t planes = weight_real.size(-2);
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(contrib_real.device());
    auto grad_c_real = torch::empty({spread, paths}, opt);
    auto grad_c_imag = torch::empty({spread, paths}, opt);
    auto grad_rates = torch::empty({paths}, opt);
    // Follow-on B: the transposed copy of the incoming gradient, (base,
    // planes) float2. 4.1 MB for a training step -- it exists so the tiled
    // kernel reads a cell's planes as 112 contiguous bytes instead of 28
    // separate sectors. Above 16 planes the chunked backward has its own
    // transposed-read variant (same layout), so the scratch is allocated
    // there too; the launcher falls back to planes-major reads when the
    // pointer is null.
    const bool wants_scratch = (tiled != 0) || planes > 16 || window_route;
    auto grad_scratch = wants_scratch
        ? torch::empty({base * planes * 2}, opt) : torch::empty({0}, opt);
    int64_t unit_base = 0;
    const int* active = active_planes_ptr(active_planes, per_unit_w, planes,
                                          base, &unit_base);
    const int* bins = plane_bins_ptr(plane_bins, per_unit_w, planes, weight_real);
    const double* tx_t = tx_times_ptr(tx_times, bins, n_tx);
    const int* table = bin_to_plane_ptr(bin_to_plane, per_unit_w, loops, weight_real, taps);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    mm25v5::launch_doppler_plane_splat_backward(
        grad_out_real.contiguous().data_ptr<float>(),
        grad_out_imag.contiguous().data_ptr<float>(),
        contrib_real.data_ptr<float>(), contrib_imag.data_ptr<float>(),
        flat_indices.data_ptr<int>(), tx_indices.data_ptr<int64_t>(),
        path_rates.data_ptr<float>(), times.data_ptr<float>(),
        weight_real.data_ptr<float>(), weight_imag.data_ptr<float>(),
        (float)phase_const,
        grad_c_real.data_ptr<float>(), grad_c_imag.data_ptr<float>(),
        grad_rates.data_ptr<float>(),
        wants_scratch ? grad_scratch.data_ptr<float>() : nullptr,
        has_amps ? forward_amps.data_ptr<float>() : nullptr,
        spread, paths, loops, n_tx, base, planes, stream.stream(),
        per_unit_w ? unit_of_path->data_ptr<int64_t>() : nullptr, active,
        bins, (int)windowed, (int)anchor, tx_t, table, (int)taps);
    return {grad_c_real, grad_c_imag, grad_rates};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "HuPR CUDA kernels";
    m.def("hupr_static_gain_splat_forward", &hupr_static_gain_splat_forward,
          "HuPR K1 forward: gain-weighted static coherent scatter");
    m.def("hupr_static_gain_splat_backward", &hupr_static_gain_splat_backward,
          "HuPR K1 backward: grads to gains and contributions");
    m.def("hupr_slow_time_splat_forward", &hupr_slow_time_splat_forward,
          "HuPR K2 forward: fused slow-time velocity splat");
    m.def("hupr_slow_time_splat_backward", &hupr_slow_time_splat_backward,
          "HuPR K2 backward: grads to contributions and path rates");
    m.def("hupr_doppler_plane_splat_forward", &hupr_doppler_plane_splat_forward,
          "HuPR K2b forward: splat into the requested Doppler bins only");
    m.def("hupr_doppler_plane_splat_backward", &hupr_doppler_plane_splat_backward,
          "HuPR K2b backward: grads to contributions and path rates",
          py::arg("grad_out_real"), py::arg("grad_out_imag"),
          py::arg("contrib_real"), py::arg("contrib_imag"),
          py::arg("flat_indices"), py::arg("tx_indices"), py::arg("path_rates"),
          py::arg("times"), py::arg("weight_real"), py::arg("weight_imag"),
          py::arg("phase_const"), py::arg("base"), py::arg("tiled") = 1,
          py::arg("forward_amps") = torch::Tensor(),
          py::arg("unit_of_path") = c10::nullopt,
          py::arg("active_planes") = c10::nullopt,
          py::arg("plane_bins") = c10::nullopt,
          py::arg("windowed") = 0, py::arg("anchor") = 0,
          py::arg("tx_times") = c10::nullopt,
          py::arg("bin_to_plane") = c10::nullopt, py::arg("taps") = 0);
    m.def("hupr_rae_spectrum_input", &hupr_rae_spectrum_input,
          "the RAE view's 2D-FFT input, built straight from the channels (bitwise)");
    m.def("hupr_rae_spectrum_input_backward", &hupr_rae_spectrum_input_backward,
          "its backward: a permutation times the slot weight");
    m.def("hupr_rae_plane_term_forward", &hupr_rae_plane_term_forward,
          "the elevation term's plane part: the pointwise chain in one pass (bitwise)");
    m.def("hupr_rae_plane_term_backward", &hupr_rae_plane_term_backward,
          "its backward, in autograd's operation order");
    m.def("hupr_static_gain_map_forward", &hupr_static_gain_map_forward,
          "HuPR K1 forward through the scene's fixed sparse map");
    m.def("hupr_static_gain_map_backward", &hupr_static_gain_map_backward,
          "HuPR K1 backward through the fixed sparse map, one warp per point");
    m.def("hupr_doppler_plane_sorted_forward", &hupr_doppler_plane_sorted_forward,
          "the sorted Doppler route's forward: contributions synthesised per cell run");
    m.def("hupr_doppler_plane_sorted_backward", &hupr_doppler_plane_sorted_backward,
          "the sorted Doppler route's backward: carrier and rate gradients per entry");
    m.def("hupr_static_scene_frames_forward", &hupr_static_scene_frames_forward,
          "K1 for a batch of frames from the BSDF carrier: the contributions "
          "are synthesised inside the fixed-order segmented sum");
    m.def("hupr_static_scene_frames_backward", &hupr_static_scene_frames_backward,
          "the carrier gradient of hupr_static_scene_frames_forward, one thread per path");
    m.def("hupr_doppler_plane_gather_forward", &hupr_doppler_plane_gather_forward,
          "HuPR K2b forward as a gather: one block per output cell, no "
          "atomics on the output, optional fused velocity-source axis",
          py::arg("contrib_real"), py::arg("contrib_imag"),
          py::arg("flat_indices"), py::arg("tx_indices"), py::arg("path_rates"),
          py::arg("times"), py::arg("weight_real"), py::arg("weight_imag"),
          py::arg("phase_const"), py::arg("base"),
          py::arg("deterministic") = 0,
          py::arg("unit_of_path") = c10::nullopt,
          py::arg("active_planes") = c10::nullopt,
          py::arg("plane_bins") = c10::nullopt,
          py::arg("windowed") = 0, py::arg("anchor") = 0,
          py::arg("tx_times") = c10::nullopt,
          py::arg("bin_to_plane") = c10::nullopt, py::arg("taps") = 0);
}
