#pragma once

#include <bcl/cl.h>
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
