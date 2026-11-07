#pragma once

#include <bcl/env.h>
#include <bcl/tensor.h>
#include <bcl/error.h>

BclTensor * bcl_tensor_add        (const BclEnv *env, const BclTensor *a, const BclTensor *b, BclTensor *c, BclError *out_err);
BclTensor * bcl_tensor_add_scalar (const BclEnv *env, const BclTensor *a, float s, BclTensor *c, BclError *out_err);
BclTensor * bcl_tensor_sum        (const BclEnv *env, const BclTensor *a, size_t axis, BclTensor *c, BclError *out_err);
