#include <bcl/tensor_ops.h>
#include <assert.h>

BclError
bcl_tensor_add (BclEnv *env, BclTensor *a, BclTensor *b, BclTensor *c)
{
	assert (env != NULL);
	assert (a != NULL && b != NULL && c != NULL);
	assert (bcl_tensor_size (a) == bcl_tensor_size (b)
			&& bcl_tensor_size (b) == bcl_tensor_size (c));

	BclError err = BCL_SUCCESS;
	cl_int cl_err = CL_SUCCESS;

	cl_kernel kernel = NULL;
	cl_mem data_a = NULL;
	cl_mem data_b = NULL;
	cl_mem data_c = NULL;

	size_t global_work_size = 0;
	cl_uint n = 0;

	kernel = bcl_env_kernel (env, BCL_KERNEL_ADD);
	global_work_size = bcl_tensor_size (a);
	n = bcl_tensor_size (a);

	data_a = bcl_tensor_data (a);
	data_b = bcl_tensor_data (b);
	data_c = bcl_tensor_data (c);

	clSetKernelArg (kernel, 0, sizeof (cl_uint), &n);
	clSetKernelArg (kernel, 1, sizeof (cl_mem), &data_a);
	clSetKernelArg (kernel, 2, sizeof (cl_mem), &data_b);
	clSetKernelArg (kernel, 3, sizeof (cl_mem), &data_c);

	cl_err = clEnqueueNDRangeKernel (bcl_env_queue (env), kernel,
			1, NULL, &global_work_size, NULL, 0, NULL, NULL);

	if (cl_err != CL_SUCCESS)
		err = cl_err;

	return err;
}
