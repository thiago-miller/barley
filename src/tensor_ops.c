#include <bcl/tensor_ops.h>
#include <assert.h>

BclError
bcl_tensor_add (BclEnv *env, BclTensor *a, BclTensor *b, BclTensor *c)
{
	assert (env != NULL);
	assert (a != NULL && b != NULL && c != NULL);
	assert (a->size == b->size && b->size == c->size);

	BclError err = BCL_SUCCESS;
	cl_int cl_err = CL_SUCCESS;

	cl_kernel kernel = NULL;
	size_t global_work_size = 0;
	cl_uint n = 0;

	kernel = env->kernels[BCL_KERNEL_ADD];
	global_work_size = a->size;
	n = a->size;

	clSetKernelArg (kernel, 0, sizeof (cl_uint), &n);
	clSetKernelArg (kernel, 1, sizeof (cl_mem), &a);
	clSetKernelArg (kernel, 2, sizeof (cl_mem), &b);
	clSetKernelArg (kernel, 3, sizeof (cl_mem), &c);

	cl_err = clEnqueueNDRangeKernel (env->queue, kernel, 1, NULL,
			&global_work_size, NULL, 0, NULL, NULL);

	if (cl_err != CL_SUCCESS)
		err = cl_err;

	return err;
}
