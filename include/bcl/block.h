#pragma once

#include <bcl/cl.h>
#include <bcl/env.h>
#include <bcl/error.h>
#include <stdlib.h>

typedef struct _BclBlock BclBlock;

struct _BclBlock
{
	cl_mem_flags flags;
	cl_mem       data;
	size_t       size;
};

static inline cl_mem_flags
bcl_block_flags (const BclBlock *block)
{
	return block != NULL ? block->flags : 0;
}

static inline cl_mem
bcl_block_data (const BclBlock *block)
{
	return block != NULL ? block->data : NULL;
}

static inline size_t
bcl_block_size (const BclBlock *block)
{
	return block != NULL ? block->size : 0;
}

BclBlock * bcl_block_alloc (BclEnv *env, float *data, size_t size, int ro, BclError *out_err);
void       bcl_block_free  (BclBlock *block);
float    * bcl_block_read  (BclBlock *block, BclEnv *env, size_t offset, size_t size, BclError *out_err);
BclError   bcl_block_write (BclBlock *block, BclEnv *env, float *data, size_t offset, size_t size);
