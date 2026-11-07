#pragma once

#include <errno.h>

// Positive error - Std errno
// Negative error - OpenCL errno
// > BCL_ERROR_BASE - Barley error
#define BCL_SUCCESS 0
#define BCL_ERROR_BASE 0x10000

typedef int BclError;

enum _BclError
{
	BCL_ERROR_INVALID_ARGUMENT = BCL_ERROR_BASE + 1,
	BCL_ERROR_SHAPE_MISMATCH,
	BCL_ERROR_SIZE_OVERFLOW,
	BCL_ERROR_ALIASING,
	BCL_ERROR_LAST
};

const char * bcl_strerror (BclError err);
