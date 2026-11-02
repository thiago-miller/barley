#pragma once

#include <bcl/error.h>
#include <stdio.h>

char * bcl_file_slurp (const char *filename, size_t *out_size, BclError *out_err);
