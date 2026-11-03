#pragma once

#include <errno.h>

// Positive error - Std errno
// Negative error - OpenCL errno
#define BCL_SUCCESS 0

typedef int BclError;

const char * bcl_strerror (BclError err);
