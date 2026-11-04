#include <bcl/error.h>
#include <bcl/cl.h>
#include <bcl/cl_error.h>
#include <string.h>

const char *
bcl_strerror (BclError err)
{
	if (err > 0)
		return strerror (err);
	else if (err < 0)
		return clGetErrorString (err);
	else
		return "Success";

	return "How did I get here?";
}
