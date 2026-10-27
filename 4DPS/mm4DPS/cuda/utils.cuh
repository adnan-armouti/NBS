// HuPR device helpers. Ours, not mm3DPS's.
//
// This header exists because hupr_kernels.cu used to #include upstream's
// utils.cuh for exactly one symbol -- sincosf_reduced -- which we wrote and
// had put into their header. It appears nowhere in
// template/reference_code/3DPS_submission. Nothing else in the HuPR kernels
// comes from that header: checked symbol by symbol, functions and macros both.
#pragma once

#include <cuda_runtime.h>

#include "sincos_reduced.cuh"
