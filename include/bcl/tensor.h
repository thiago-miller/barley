#pragma once

#include <bcl/env.h>
#include <bcl/block.h>
#include <bcl/error.h>
#include <stdlib.h>

typedef struct _BclTensor BclTensor;

struct _BclTensor
{
	size_t    rank;
	size_t   *shape;
	size_t   *strides;
	size_t    offset;
	size_t    size;
	int       owner;
	BclBlock *block;
};

BclTensor * bcl_tensor_alloc (BclEnv *env, size_t rank, size_t *shape, float *data, int ro, BclError *out_err);
void        bcl_tensor_free  (BclTensor *tensor);
