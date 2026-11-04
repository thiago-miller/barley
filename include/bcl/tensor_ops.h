#pragma once

#include <bcl/env.h>
#include <bcl/tensor.h>
#include <bcl/error.h>

BclError bcl_tensor_add (BclEnv *env, BclTensor *a, BclTensor *b, BclTensor *c);
