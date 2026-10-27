"""Native CUDA kernel for the lightweight multibounce CSI render.

Matches the verified reference (method/multibounce_torch.render_csi[_vec], which matches mmIR's
image_method.py to machine precision). The kernel parallelizes over paths: each thread evaluates
one path's Fresnel product + 1/L, then atomic-accumulates its contribution into the complex CSI
across all subcarriers. Geometry (path lengths, incidence angles, plane indices) is precomputed
(`flatten_paths`); this kernel is the hot render that runs every optimization step.

Built on first use via torch.utils.cpp_extension.load_inline (CUDA 11.8, sm_89).
"""
import math
import torch
from torch.utils.cpp_extension import load_inline

C_LIGHT = 299792458.0

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ double fresnel_te(double eps, double cos_i){
    cos_i = fmin(fmax(cos_i, 1e-3), 1.0);
    double s = sqrt(fmax(eps - (1.0 - cos_i*cos_i), 1e-9));
    return (cos_i - s) / (cos_i + s + 1e-12);
}
// d(fresnel_te)/d(eps) = -cos / ( s * (cos+s)^2 ),  s = sqrt(eps - sin^2)
__device__ __forceinline__ double dfresnel_te(double eps, double cos_i){
    cos_i = fmin(fmax(cos_i, 1e-3), 1.0);
    double s = sqrt(fmax(eps - (1.0 - cos_i*cos_i), 1e-9));
    double cps = cos_i + s;
    return -cos_i / (s * cps * cps + 1e-12);
}

__global__ void mb_kernel(
    const long*  __restrict__ link, const double* __restrict__ L, const long* __restrict__ nb,
    const long*  __restrict__ p0,   const double* __restrict__ c0,
    const long*  __restrict__ p1,   const double* __restrict__ c1,
    const double* __restrict__ eps, const double* __restrict__ sc,
    int n_paths, int n_sc, double c_light, double* __restrict__ Hout /* (n_links,n_sc,2) */)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_paths) return;
    int li = (int)link[idx];
    double Lp = L[idx];
    double gamma = 1.0;
    if (nb[idx] >= 1) gamma *= fresnel_te(eps[p0[idx]], c0[idx]);
    if (nb[idx] >= 2) gamma *= fresnel_te(eps[p1[idx]], c1[idx]);
    double amp = gamma / Lp;
    for (int s = 0; s < n_sc; ++s) {
        double ph = 2.0 * M_PI * sc[s] * Lp / c_light;     // exp(-j ph): re=cos, im=-sin
        double* base = Hout + (li * n_sc + s) * 2;
        atomicAdd(base + 0,  amp * cos(ph));
        atomicAdd(base + 1, -amp * sin(ph));
    }
}

// backward: grad_eps[k] = sum over paths through plane k, over subcarriers, of
//   2 * Re( conj(grad_H) * dA/deps[k] ),  dA/deps = (dgamma/deps / L) exp(-j ph)
// PyTorch passes grad_H = dL/dRe(H) + i dL/dIm(H) (full real gradient, no 1/2 factor).
__global__ void mb_kernel_bwd(
    const long*  __restrict__ link, const double* __restrict__ L, const long* __restrict__ nb,
    const long*  __restrict__ p0,   const double* __restrict__ c0,
    const long*  __restrict__ p1,   const double* __restrict__ c1,
    const double* __restrict__ eps, const double* __restrict__ sc,
    const double* __restrict__ gH /* (n_links,n_sc,2) = grad wrt H */,
    int n_paths, int n_sc, double c_light, double* __restrict__ grad_eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_paths) return;
    int li = (int)link[idx]; double Lp = L[idx]; int n = (int)nb[idx];
    int k0 = (int)p0[idx], k1 = (int)p1[idx];
    double g0 = (n >= 1) ? fresnel_te(eps[k0], c0[idx]) : 1.0;
    double g1 = (n >= 2) ? fresnel_te(eps[k1], c1[idx]) : 1.0;
    // sum_s 2*(gH_re*Re(exp(-jph)/L) + gH_im*Im(exp(-jph)/L)); material-independent factor
    double sdot = 0.0;
    for (int s = 0; s < n_sc; ++s) {
        double ph = 2.0 * M_PI * sc[s] * Lp / c_light;
        const double* gb = gH + (li * n_sc + s) * 2;
        sdot += gb[0] * (cos(ph) / Lp) + gb[1] * (-sin(ph) / Lp);
    }
    if (n >= 1) atomicAdd(&grad_eps[k0], dfresnel_te(eps[k0], c0[idx]) * g1 * sdot);
    if (n >= 2) atomicAdd(&grad_eps[k1], g0 * dfresnel_te(eps[k1], c1[idx]) * sdot);
}

torch::Tensor mb_forward(torch::Tensor link, torch::Tensor L, torch::Tensor nb,
                         torch::Tensor p0, torch::Tensor c0, torch::Tensor p1, torch::Tensor c1,
                         torch::Tensor eps, torch::Tensor sc, int64_t n_links, double c_light){
    int n_paths = link.size(0); int n_sc = sc.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(link.device());
    auto Hout = torch::zeros({n_links, n_sc, 2}, opts);
    int threads = 256, blocks = (n_paths + threads - 1) / threads;
    mb_kernel<<<blocks, threads>>>(
        link.data_ptr<long>(), L.data_ptr<double>(), nb.data_ptr<long>(),
        p0.data_ptr<long>(), c0.data_ptr<double>(), p1.data_ptr<long>(), c1.data_ptr<double>(),
        eps.data_ptr<double>(), sc.data_ptr<double>(), n_paths, n_sc, c_light, Hout.data_ptr<double>());
    return Hout;
}

torch::Tensor mb_backward(torch::Tensor link, torch::Tensor L, torch::Tensor nb,
                          torch::Tensor p0, torch::Tensor c0, torch::Tensor p1, torch::Tensor c1,
                          torch::Tensor eps, torch::Tensor sc, torch::Tensor gH,
                          int64_t n_planes, double c_light){
    int n_paths = link.size(0); int n_sc = sc.size(0);
    auto grad_eps = torch::zeros({n_planes}, eps.options());
    int threads = 256, blocks = (n_paths + threads - 1) / threads;
    mb_kernel_bwd<<<blocks, threads>>>(
        link.data_ptr<long>(), L.data_ptr<double>(), nb.data_ptr<long>(),
        p0.data_ptr<long>(), c0.data_ptr<double>(), p1.data_ptr<long>(), c1.data_ptr<double>(),
        eps.data_ptr<double>(), sc.data_ptr<double>(), gH.data_ptr<double>(),
        n_paths, n_sc, c_light, grad_eps.data_ptr<double>());
    return grad_eps;
}
"""

_CPP_SRC = ("torch::Tensor mb_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
            "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double);\n"
            "torch::Tensor mb_backward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
            "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double);")

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(name="mb_cuda", cpp_sources=_CPP_SRC, cuda_sources=_CUDA_SRC,
                           functions=["mb_forward", "mb_backward"], verbose=False,
                           extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"])
    return _EXT


def render_csi_cuda(flat, eps, sc_hz):
    """Complex CSI (n_links, n_sc) via the native CUDA kernel (forward only). flat from flatten_paths(..., 'cuda')."""
    sc = torch.as_tensor(sc_hz, dtype=torch.float64, device="cuda").contiguous()
    H = _ext().mb_forward(flat["link"].contiguous(), flat["L"].contiguous(), flat["nb"].contiguous(),
                          flat["p0"].contiguous(), flat["c0"].contiguous(), flat["p1"].contiguous(),
                          flat["c1"].contiguous(), eps.double().contiguous(), sc,
                          int(flat["n_links"]), C_LIGHT)
    return torch.complex(H[..., 0], H[..., 1])


class _MBFunction(torch.autograd.Function):
    """Fully-fused CUDA forward + backward; differentiable in eps (per-plane permittivity)."""
    @staticmethod
    def forward(ctx, eps, link, L, nb, p0, c0, p1, c1, sc, n_links, n_planes):
        Hout = _ext().mb_forward(link, L, nb, p0, c0, p1, c1, eps.double().contiguous(), sc, n_links, C_LIGHT)
        ctx.save_for_backward(eps, link, L, nb, p0, c0, p1, c1, sc)
        ctx.n_planes = n_planes
        return torch.complex(Hout[..., 0], Hout[..., 1])

    @staticmethod
    def backward(ctx, grad_H):
        eps, link, L, nb, p0, c0, p1, c1, sc = ctx.saved_tensors
        gH = torch.stack([grad_H.real, grad_H.imag], dim=-1).double().contiguous()
        grad_eps = _ext().mb_backward(link, L, nb, p0, c0, p1, c1, eps.double().contiguous(), sc, gH,
                                      ctx.n_planes, C_LIGHT)
        return (grad_eps, None, None, None, None, None, None, None, None, None, None)


def render_csi_cuda_diff(flat, eps, sc_hz):
    """Differentiable CSI via the fused CUDA forward+backward kernels (grad flows to eps)."""
    sc = torch.as_tensor(sc_hz, dtype=torch.float64, device="cuda").contiguous()
    return _MBFunction.apply(eps, flat["link"].contiguous(), flat["L"].contiguous(), flat["nb"].contiguous(),
                             flat["p0"].contiguous(), flat["c0"].contiguous(), flat["p1"].contiguous(),
                             flat["c1"].contiguous(), sc, int(flat["n_links"]), int(eps.shape[0]))
