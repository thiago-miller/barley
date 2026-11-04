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

static inline size_t
bcl_tensor_rank (const BclTensor *tensor)
{
	return tensor != NULL ? tensor->rank : 0;
}

static inline const size_t *
bcl_tensor_shape (const BclTensor *tensor)
{
	return tensor != NULL ? tensor->shape : NULL;
}

static inline size_t
bcl_tensor_size (const BclTensor *tensor)
{
	return tensor != NULL ? tensor->size : 0;
}

static inline cl_mem
bcl_tensor_data (const BclTensor *tensor)
{
	return (tensor != NULL && tensor->block != NULL) ? tensor->block->data : NULL;
}

BclTensor * bcl_tensor_alloc (BclEnv *env, size_t rank, size_t *shape, float *data, int ro, BclError *out_err);
void        bcl_tensor_free  (BclTensor *tensor);
