// Macros for neatly error checking OpenCL API functions
// Source: https://gist.github.com/allanmac/9328bb2d6a99b86883195f8f78fd1b93
// Author: allanmac

#pragma once

#include <bcl/cl.h>

char const * clGetErrorString(cl_int const err);
