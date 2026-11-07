#include <bcl/error.h>
#include <bcl/cl.h>
#include <bcl/cl_error.h>
#include <string.h>

static const char *const bcl_messages[] =
{
	[BCL_ERROR_INVALID_ARGUMENT - BCL_ERROR_BASE] = "Invalid argument",
	[BCL_ERROR_SHAPE_MISMATCH   - BCL_ERROR_BASE] = "Tensor shapes are incompatible",
	[BCL_ERROR_SIZE_OVERFLOW    - BCL_ERROR_BASE] = "Tensor dimension or size exceeds the 32-bit limit",
	[BCL_ERROR_ALIASING         - BCL_ERROR_BASE] = "Output tensor shares memory with an input"
};

const char *
bcl_strerror (BclError err)
{
	const char *msg = NULL;

	if (err == BCL_SUCCESS)
		return "Success";

	if (err >= BCL_ERROR_BASE)
		{
			if (err < BCL_ERROR_LAST)
				msg = bcl_messages[err - BCL_ERROR_BASE];
			return msg != NULL ? msg : "Unknown bcl error";
		}

	if (err > 0)
		return strerror (err);

	msg = clGetErrorString (err);
	return msg != NULL ? msg : "Unknown OpenCL error";
}
