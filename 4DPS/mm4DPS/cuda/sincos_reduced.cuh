// The ONE definition of sincosf_reduced.
//
// It is ours -- it appears nowhere in
// template/reference_code/3DPS_submission -- and it has two callers in two
// separately built extensions: mm4DPS/cuda/kernels.cu (15 calls) and the
// vendored mm4DPS/splat/cuda/scatter_splat.cu (2 calls, where we
// replaced upstream's plain sincosf). It lived in both trees' utils.cuh,
// byte-identical, until 2026-09-03.
//
// It sits here, on our side, rather than in the vendored header, because the
// symbol is ours: the vendored build adds this directory to its include path.
// That points a dependency from the vendored tree at ours, which is the right
// way round -- the alternative would put our kernels back inside theirs.
//
// Declared in namespace mm25v5 so both callers resolve it the way they always
// did: unqualified from inside the namespace, and mm25v5:: from outside.
#pragma once

#include <cuda_runtime.h>

namespace mm25v5 {

// Accurate sincos for large carrier phases under --use_fast_math.
// Step-5 phases are 2*pi*f0*tau ~ 1e4 rad; fast-math sincosf loses range
// reduction there (measured: up to 24% field deviation in dense speckle
// after coherent cancellation). Reducing in fp64 first is exact for any
// fp32 input and leaves the actual sin/cos evaluation on a small argument
// where the fast intrinsic is accurate to ~1e-7.
//
// Phase 5 W1b: the reduction multiplies by 1/2*pi instead of dividing by
// 2*pi. The comment above used to say this "costs three fp64 ops per path";
// measured, the fp64 *divide* was the whole cost -- fp64 runs at 1/64 rate on
// Ada and a divide is ~20 ops, so it was 0.484 ms of the 20.5 M reductions a
// training step's K2b forward does, and the backward does two of those per
// path. Multiplying by the reciprocal is 0.260 ms for the same 20.5 M.
//
// It is not merely close, it is the same number. Both quotients are within one
// double ulp of the exact p/2*pi, so their floors can differ only if the exact
// quotient lies within ~2 ulps of an integer -- for |phi| <= 1e4 that is a
// ~2e-12 wide window against a float32 spacing of ~1e-3 near that magnitude,
// i.e. ~2e-9 of inputs. And when they do differ the reduced angle shifts by
// exactly 2*pi, so sinf changes by at most sin(2*pi) ~ 2.4e-7 absolute and
// cosf by ~0 -- one term of a 64-term sum, under the measured 1e-7 relative
// nondeterminism floor of the accumulated quantity.
//
// Measured, not argued: bitwise identical over the 20.5 M (path, loop) phases
// of a real training step and the 46.9 M of a real evaluation chunk, and over
// the whole K2b forward and backward on both.
__device__ __forceinline__ void sincosf_reduced(float phi, float* s, float* c) {
    const double TWO_PI = 6.283185307179586476925286766559;
    const double INV_TWO_PI = 0.159154943091895335768883763372514;
    const double p = (double)phi;
    const float reduced = (float)(p - TWO_PI * floor(p * INV_TWO_PI));
    sincosf(reduced, s, c);
}

}  // namespace mm25v5
