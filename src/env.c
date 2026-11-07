#include <bcl/env.h>
#include <bcl/utils.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BCL_KERNEL_PATH "kernels/tensor_ops.cl"

static const char *kernel_names[BCL_KERNEL_COUNT] = {
	[BCL_KERNEL_ADD]           = "tensor_add",
	[BCL_KERNEL_ADD_SCALAR]    = "tensor_add_scalar",
	[BCL_KERNEL_ADD_BROADCAST] = "tensor_add_broadcast",
	[BCL_KERNEL_SUM]           = "tensor_sum"
};

static cl_device_id
bcl_get_device_id (cl_device_type device_type, BclError *out_err)
{
	BclError err = BCL_SUCCESS;
	cl_int cl_err = CL_SUCCESS;

	cl_uint num_platforms = 0;
	cl_platform_id *platforms = NULL;
	cl_device_id device_id = NULL;

	clGetPlatformIDs (0, NULL, &num_platforms);
	if (num_platforms == 0)
		{
			err = CL_INVALID_PLATFORM;
			goto RET;
		}

	platforms = malloc (sizeof (cl_platform_id) * num_platforms);
	if (platforms == NULL)
		{
			err = errno;
			goto RET;
		}

	clGetPlatformIDs (num_platforms, platforms, NULL);

	for (cl_uint i = 0; i < num_platforms; i++)
		{
			cl_uint num_devices = 0;

			cl_err = clGetDeviceIDs (platforms[i], device_type, 1, &device_id, &num_devices);

			if (cl_err == CL_SUCCESS && num_devices > 0)
				{
					err = cl_err;
					break;
				}
		}

	if (device_id == NULL)
		{
			err = CL_DEVICE_NOT_FOUND;
			goto RET;
		}

RET:
	if (platforms != NULL)
		free (platforms);

	if (out_err != NULL)
		*out_err = err;

	return device_id;
}

static BclError
bcl_create_kernels (cl_kernel *kernels, cl_program program)
{
	BclError err = BCL_SUCCESS;
	cl_int cl_err = CL_SUCCESS;

	for (int i = 0; i < BCL_KERNEL_COUNT; i++)
		{
			kernels[i] = clCreateKernel (program, kernel_names[i], &cl_err);
			if (cl_err != CL_SUCCESS)
				{
					err = cl_err;

					for (int j = i - 1; j >= 0; j--)
						clReleaseKernel (kernels[j]);

					break;
				}
		}

	return err;
}

BclEnv *
bcl_env_alloc (cl_device_type device_type, BclError *out_err)
{
	BclEnv *env = NULL;

	BclError err = BCL_SUCCESS;
	cl_int cl_err = CL_SUCCESS;

	cl_device_id device_id;
	cl_context context = NULL;
	cl_command_queue queue = NULL;
	cl_program program = NULL;
	cl_kernel kernels[BCL_KERNEL_COUNT] = {0};

	const char *kernel_path = NULL;
	char *kernel_source = NULL;
	size_t log_size = 0;
	char *log = NULL;

	device_id = bcl_get_device_id (device_type, &err);
	if (err != BCL_SUCCESS)
		goto CLEAN;

	context = clCreateContext (NULL, 1, &device_id, NULL, NULL, &cl_err);
	if (cl_err != CL_SUCCESS)
		{
			err = cl_err;
			goto CLEAN;
		}

	queue = clCreateCommandQueueWithProperties (context, device_id, NULL, &cl_err);
	if (cl_err != CL_SUCCESS)
		{
			err = cl_err;
			goto CLEAN;
		}

	kernel_path = getenv ("BCL_KERNEL_PATH");
	if (kernel_path == NULL)
		kernel_path = BCL_KERNEL_PATH;

	kernel_source = bcl_file_slurp (kernel_path, NULL, &err);
	if (err != BCL_SUCCESS)
		goto CLEAN;

	program = clCreateProgramWithSource (context, 1, (const char **) &kernel_source, NULL, &cl_err);
	if (cl_err != CL_SUCCESS)
		{
			err = cl_err;
			goto CLEAN;
		}

	cl_err = clBuildProgram (program, 1, &device_id, NULL, NULL, NULL);
	if (cl_err != CL_SUCCESS)
		{
			clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			log = malloc (log_size * sizeof (char));
			clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
			fprintf (stderr, "Kernel Build Error:\n%s\n", log);
			abort ();
		}

	err = bcl_create_kernels (kernels, program);
	if (err != BCL_SUCCESS)
		goto CLEAN;

	env = malloc (sizeof (BclEnv));
	if (env == NULL)
		{
			err = errno;
			goto CLEAN;
		}

	*env = (BclEnv) {
		.device_type = device_type,
		.device_id   = device_id,
		.context     = context,
		.queue       = queue,
		.program     = program
	};

	memcpy (env->kernels, kernels, sizeof (kernels));

RET:
	if (kernel_source != NULL)
		free (kernel_source);

	if (out_err != NULL)
		*out_err = err;

	return env;

CLEAN:
	if (program != NULL)
		clReleaseProgram (program);

	if (queue != NULL)
		clReleaseCommandQueue (queue);

	if (context != NULL)
		clReleaseContext (context);

	goto RET;
}

void
bcl_env_free (BclEnv *env)
{
	if (env == NULL)
		return;

	for (int i = 0; i < BCL_KERNEL_COUNT; i++)
		clReleaseKernel (env->kernels[i]);

	clReleaseProgram (env->program);
	clReleaseCommandQueue (env->queue);
	clReleaseContext (env->context);

	free (env);
}
