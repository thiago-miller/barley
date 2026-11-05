#pragma once

#include <bcl/env.h>
#include <bcl/tensor.h>
#include <bcl/error.h>

BclError bcl_tensor_add        (const BclEnv *env, const BclTensor *a, const BclTensor *b, BclTensor *c);
BclError bcl_tensor_add_scalar (const BclEnv *env, const BclTensor *a, float s, BclTensor *c);
