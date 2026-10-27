// The closed-form Doppler-plane amplitude of one path: the Hann Doppler PSF.
//
// A path's slow-time signal is one tone, exp(j (phi0 + omega l)) over the L
// TDM loops, so its amplitude in Doppler plane d -- the plane weights of
// signal_model.doppler_dft_rows applied to it -- is not a sum over the 64
// loops but the window's response at the path's fractional Doppler bin:
//
//   nu  = omega L / 2 pi                       (the path's unshifted Doppler bin)
//   D(delta) = sum_l exp(-2 pi j delta l / L)  (the Dirichlet kernel; L-periodic)
//            = exp(-j pi delta (L-1)/L) * sin(pi delta) / sin(pi delta / L)
//   A_d = exp(j phi0) [ 0.5 D(d'-nu) - 0.25 D(d'-1-nu) - 0.25 D(d'+1-nu) - m_d D(-nu) ]
//
// with d' the plane's unshifted frequency, the three Dirichlet terms the
// periodic Hann's three-bin DFT (rasterizer.hann_psf, the range PSF, is the
// same expression), and m_d the folded slow-time mean removal (0.5 at DC,
// -0.25 at +/-1, 0 elsewhere). Without a window: A_d = exp(j phi0) D(d'-nu).
// The static anchor (plane 0) is exp(j phi0) (loop 0) or exp(j phi0) D(-nu)/L
// (the dc mean). The float64 reference is
// signal_model.doppler_plane_amplitudes_reference; the two agree to 4e-14
// there, and the fp32 code here is gated against that reference.
//
// Precision, fp32. Everything that can lose digits is arranged not to:
//   * nu and phi0 are formed in fp64 from the rate and split into an integer
//     bin n and a fraction f (|f| <= 1/2); every plane's offset is then the
//     exact integer m = d' - n (reduced mod L: D is L-periodic) minus f, so
//     the small quantities near a bin centre are never differences of
//     large ones.
//   * sin(pi delta) = -(-1)^m sin(pi f), cos(pi delta) = (-1)^m cos(pi f): the
//     sines over the full range are one small-argument evaluation per path.
//   * sin/cos of the small arguments (|x| <= 1.6) are odd/even polynomials,
//     relative accuracy ~1e-7 down to x = 0 (the build is IEEE, no fast
//     math; the polynomials are cheap and keep the small-argument relative
//     accuracy that sin(pi delta/L) near a bin needs).
//   * R = sin(pi delta)/sin(pi delta/L) and its derivative R' come from a
//     series in delta for |delta| < 1/2 (no 0/0, no cancellation) and from
//     the direct formula beyond, where the cancellation in R' costs at most
//     a factor 1.2.
#pragma once

#include <cuda_runtime.h>
#include <math.h>

namespace mm25v5 {

// sin and cos for |x| <= 1.7, odd/even Taylor polynomials; relative error
// ~1e-7 at 1.6 and exact to first order at 0.
__device__ __forceinline__ float sin_small(float x) {
    const float x2 = x * x;
    // x - x^3/6 + x^5/120 - x^7/5040 + x^9/362880 - x^11/39916800 + x^13/6227020800
    float p = 1.6059044e-10f;
    p = fmaf(p, x2, -2.5052108e-08f);
    p = fmaf(p, x2, 2.7557319e-06f);
    p = fmaf(p, x2, -1.9841270e-04f);
    p = fmaf(p, x2, 8.3333333e-03f);
    p = fmaf(p, x2, -1.6666667e-01f);
    p = fmaf(p, x2, 1.0f);
    return x * p;
}

__device__ __forceinline__ float cos_small(float x) {
    const float x2 = x * x;
    // 1 - x^2/2 + x^4/24 - x^6/720 + x^8/40320 - x^10/3628800 + x^12/479001600 - x^14/87178291200
    float p = -1.1470746e-11f;
    p = fmaf(p, x2, 2.0876757e-09f);
    p = fmaf(p, x2, -2.7557319e-07f);
    p = fmaf(p, x2, 2.4801587e-05f);
    p = fmaf(p, x2, -1.3888889e-03f);
    p = fmaf(p, x2, 4.1666667e-02f);
    p = fmaf(p, x2, -5.0000000e-01f);
    p = fmaf(p, x2, 1.0f);
    return p;
}

// P(z) = sum_{k=0}^{8} (-1)^k z^k / (2k+1)!  -- sinc(x) with z = x^2 -- and dP/dz.
// Nine terms: at z = 2.47 (|delta| = 1/2) the tenth is below 1e-8.
__device__ __forceinline__ float sinc_poly(float z, float* dP) {
    const float coef[9] = {
        1.0f, -1.0f / 6.0f, 1.0f / 120.0f, -1.0f / 5040.0f, 1.0f / 362880.0f,
        -1.0f / 39916800.0f, 1.0f / 6227020800.0f, -1.0f / 1307674368000.0f,
        1.0f / 355687428096000.0f};
    float p = coef[8];
    float dp = 8.0f * coef[8];
#pragma unroll
    for (int k = 7; k >= 1; --k) {
        p = fmaf(p, z, coef[k]);
        dp = fmaf(dp, z, (float)k * coef[k]);
    }
    p = fmaf(p, z, coef[0]);
    *dP = dp;
    return p;
}

// Per-path quantities of the closed form, computed once per (path, source).
struct ClosedPath {
    int n;            // round(nu)
    float f;          // nu - n, |f| <= 1/2
    float sin_pf;     // sin(pi f)
    float cos_pf;     // cos(pi f)
    float2 Ef;        // exp(j pi f (L-1)/L)
    float2 e0;        // exp(j phi0)
    float kt0;        // phase_const * t0: d phi0 / d rate
    float dnu;        // d nu / d rate = phase_const * T * L / 2 pi
};

// `tx_times` is (n_tx, 2) float64: each TX's first sample time t0 and its
// loop period T (forward_render._tx_times64) -- in double, because nu's
// slope turns a float32 T's rounding into ~6e-6 of the amplitude peak.
__device__ __forceinline__ ClosedPath closed_path(
    float rate, const double* __restrict__ tx_times, int64_t tx,
    float phase_const, int L)
{
    ClosedPath c;
    const double t0 = tx_times[2 * tx];
    const double T = tx_times[2 * tx + 1];
    const double phi0 = (double)phase_const * (double)rate * t0;
    const double nu = (double)phase_const * (double)rate * T * (double)L / (2.0 * M_PI);
    const double n = rint(nu);
    c.n = (int)n;
    c.f = (float)(nu - n);
    const float pf = (float)M_PI * c.f;
    c.sin_pf = sin_small(pf);
    c.cos_pf = cos_small(pf);
    const float pfL = pf * (float)(L - 1) / (float)L;
    c.Ef = make_float2(cos_small(pfL), sin_small(pfL));
    // phi0 reduced to [-pi, pi] in fp64, then the small-argument evaluation:
    // a unit phasor needs absolute accuracy only.
    const double phi_r = phi0 - 2.0 * M_PI * rint(phi0 / (2.0 * M_PI));
    const float pr = (float)phi_r;
    if (fabsf(pr) <= 1.7f) {
        c.e0 = make_float2(cos_small(pr), sin_small(pr));
    } else {
        // |pr| in (1.7, pi]: evaluate at pi - |pr| (small) and fold the signs.
        const float q = (float)M_PI - fabsf(pr);
        const float s = sin_small(q), co = -cos_small(q);
        c.e0 = make_float2(co, pr >= 0.0f ? s : -s);
    }
    c.kt0 = (float)((double)phase_const * t0);
    c.dnu = (float)((double)phase_const * T * (double)L / (2.0 * M_PI));
    return c;
}

// D(delta) and dD/ddelta for delta = m - f, m an integer (any), f the path's
// fraction. Returns D in *D and dD/ddelta in *dD.
__device__ __forceinline__ void dirichlet_term(
    int m, const ClosedPath& c, int L, float2* D, float2* dD)
{
    // D is L-periodic: reduce m to [-L/2, L/2).
    int mr = m % L;
    if (mr >= L / 2) mr -= L;
    if (mr < -L / 2) mr += L;
    const float delta = (float)mr - c.f;                 // |delta| <= L/2 + 1/2
    const float sign = (mr & 1) ? -1.0f : 1.0f;          // (-1)^m
    const float Lf = (float)L;
    const float pi = (float)M_PI;
    float R, Rp;
    if (fabsf(delta) < 0.5f) {
        // R = L sinc(delta) / sinc(delta/L): even polynomials in a = (pi delta)^2
        // and b = (pi delta / L)^2, nine terms (a <= 2.47: the tenth is < 1e-8).
        const float pd = pi * delta;
        const float a = pd * pd, b = a / (Lf * Lf);
        float dPa, dPb;
        const float Pa = sinc_poly(a, &dPa), Pb = sinc_poly(b, &dPb);
        R = Lf * Pa / Pb;
        // dR/ddelta = L [P'(a) 2 pi^2 delta Pb - Pa P'(b) 2 pi^2 delta / L^2] / Pb^2
        const float two_pi2_delta = 2.0f * pi * pi * delta;
        Rp = Lf * (dPa * two_pi2_delta * Pb - Pa * dPb * two_pi2_delta / (Lf * Lf)) / (Pb * Pb);
    } else {
        // sin(pi delta) = -(-1)^m sin(pi f); cos(pi delta) = (-1)^m cos(pi f)
        const float s = -sign * c.sin_pf;
        const float co = sign * c.cos_pf;
        const float x = pi * delta / Lf;                 // |x| <= 1.6
        const float sL = sin_small(x), cL = cos_small(x);
        R = s / sL;
        Rp = pi * (co * sL - s * cL / Lf) / (sL * sL);
    }
    // phase exp(-j pi delta (L-1)/L) = exp(-j pi mr (L-1)/L) exp(j pi f (L-1)/L)
    //                                = (-1)^mr exp(j pi mr / L) Ef
    const float xm = pi * (float)mr / Lf;                // |xm| <= pi/2
    const float2 em = make_float2(cos_small(xm), sin_small(xm));
    float2 ph = make_float2(em.x * c.Ef.x - em.y * c.Ef.y, em.x * c.Ef.y + em.y * c.Ef.x);
    ph.x *= sign; ph.y *= sign;
    D->x = ph.x * R; D->y = ph.y * R;
    // dD = ph (j psi' R + R'), psi' = -pi (L-1)/L
    const float psip = -pi * (Lf - 1.0f) / Lf;
    const float ure = Rp, uim = psip * R;
    dD->x = ph.x * ure - ph.y * uim;
    dD->y = ph.x * uim + ph.y * ure;
}

// The plane's amplitude and its rate derivative. `unshifted` is the plane's
// unshifted Doppler frequency d' in [0, L); `anchor` selects plane 0's
// form (0: loop 0, 1: dc mean) and is used with unshifted < 0. `fold_only`
// returns just the folded mean-removal term -m_d D(-nu) of a windowed plane
// (the 15-tap route's two fixed entries at bins 1 and L-1: the mean removal
// is not compact, every path leaks its D(-nu) there).
__device__ __forceinline__ void closed_form_plane(
    const ClosedPath& c, int L, int unshifted, int windowed, int anchor,
    float2* A, float2* dA, int fold_only = 0)
{
    float2 body, dbody;
    if (fold_only) {
        float md = 0.0f;
        if (unshifted == 0) md = 0.5f;
        else if (unshifted == 1 || unshifted == L - 1) md = -0.25f;
        float2 Dz, dDz;
        dirichlet_term(-c.n, c, L, &Dz, &dDz);
        body = make_float2(-md * Dz.x, -md * Dz.y);
        dbody = make_float2(-md * dDz.x, -md * dDz.y);
    } else if (unshifted < 0) {
        if (anchor == 0) {
            body = make_float2(1.0f, 0.0f); dbody = make_float2(0.0f, 0.0f);
        } else {
            float2 Dz, dDz;
            dirichlet_term(-c.n, c, L, &Dz, &dDz);     // delta = -n - f = -nu
            body = make_float2(Dz.x / (float)L, Dz.y / (float)L);
            dbody = make_float2(dDz.x / (float)L, dDz.y / (float)L);
        }
    } else {
        const int m = unshifted - c.n;
        float2 D0, dD0;
        dirichlet_term(m, c, L, &D0, &dD0);
        if (windowed) {
            float2 Dm, dDm, Dp, dDp;
            dirichlet_term(m - 1, c, L, &Dm, &dDm);
            dirichlet_term(m + 1, c, L, &Dp, &dDp);
            body = make_float2(0.5f * D0.x - 0.25f * (Dm.x + Dp.x),
                               0.5f * D0.y - 0.25f * (Dm.y + Dp.y));
            dbody = make_float2(0.5f * dD0.x - 0.25f * (dDm.x + dDp.x),
                                0.5f * dD0.y - 0.25f * (dDm.y + dDp.y));
            float md = 0.0f;
            if (unshifted == 0) md = 0.5f;
            else if (unshifted == 1 || unshifted == L - 1) md = -0.25f;
            if (md != 0.0f) {
                float2 Dz, dDz;
                dirichlet_term(-c.n, c, L, &Dz, &dDz);
                body.x -= md * Dz.x; body.y -= md * Dz.y;
                dbody.x -= md * dDz.x; dbody.y -= md * dDz.y;
            }
        } else {
            body = D0; dbody = dD0;
        }
    }
    // A = e0 body; dA/drate = e0 (j kt0 body - dbody dnu)
    A->x = c.e0.x * body.x - c.e0.y * body.y;
    A->y = c.e0.x * body.y + c.e0.y * body.x;
    const float gx = -c.kt0 * body.y - dbody.x * c.dnu;
    const float gy = c.kt0 * body.x - dbody.y * c.dnu;
    dA->x = c.e0.x * gx - c.e0.y * gy;
    dA->y = c.e0.x * gy + c.e0.y * gx;
}

} // namespace mm25v5
