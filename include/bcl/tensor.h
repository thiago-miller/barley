#pragma once

#include <bcl/env.h>
#include <bcl/block.h>
#include <bcl/error.h>
#include <stdlib.h>

#define BCL_TENSOR_MAX_RANK 8

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

static inline const size_t *
bcl_tensor_strides (const BclTensor *tensor)
{
	return tensor != NULL ? tensor->strides : NULL;
}

static inline size_t
bcl_tensor_size (const BclTensor *tensor)
{
	return tensor != NULL ? tensor->size : 0;
}

static inline int
bcl_tensor_is_owner (const BclTensor *tensor)
{
	return tensor != NULL ? tensor->owner : 0;
}

static inline int
bcl_tensor_is_contiguous (const BclTensor *tensor)
{
	return tensor != NULL ? 1 : 0;
}

static inline cl_mem
bcl_tensor_data (const BclTensor *tensor)
{
	return (tensor != NULL && tensor->block != NULL) ? tensor->block->data : NULL;
}

static inline float *
bcl_tensor_read (const BclTensor *tensor, const BclEnv *env, size_t offset, size_t size, BclError *out_err)
{
	return tensor != NULL ? bcl_block_read (tensor->block, env, offset, size, out_err) : NULL;
}

static inline int
bcl_tensor_write (BclTensor *tensor, const BclEnv *env, const float *data, size_t offset, size_t size)
{
	return tensor != NULL ? bcl_block_write (tensor->block, env, data, offset, size) : EINVAL;
}

BclTensor * bcl_tensor_alloc (const BclEnv *env, size_t rank, const size_t *shape, const float *data, int ro, BclError *out_err);
void        bcl_tensor_free  (BclTensor *tensor);
