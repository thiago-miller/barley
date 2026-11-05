#include <bcl/tensor_ops.h>
#include <stdio.h>
#include <assert.h>

static void
compute_broadcasting_metadata (const BclTensor *a, const BclTensor *b,
		cl_uint8 *target_shape, cl_uint8 *out_stride_a, cl_uint8 *out_stride_b)
{
	size_t size_a = 0;
	size_t size_b = 0;

	ssize_t out_idx = 0;
	ssize_t a_idx = 0;
	ssize_t b_idx = 0;

	const size_t *a_shape = NULL;
	const size_t *b_shape = NULL;
	const size_t *a_strides = NULL;
	const size_t *b_strides = NULL;

	// Initialize outputs with default values (Shape=1, Stride=0)
	for (size_t i = 0; i < BCL_TENSOR_MAX_RANK; i++)
		{
			out_stride_a->s[i] = 0;
			out_stride_b->s[i] = 0;
			target_shape->s[i] = 1;
		}

	// Process from right to left (trailing dimensions)
	out_idx = BCL_TENSOR_MAX_RANK - 1;
	a_idx = bcl_tensor_rank (a) - 1;
	b_idx = bcl_tensor_rank (b) - 1;

	// Get the shapes and strides
	a_shape = bcl_tensor_shape (a);
	b_shape = bcl_tensor_shape (b);
	a_strides =  bcl_tensor_strides (a);
	b_strides =  bcl_tensor_strides (b);

	while (a_idx >= 0 || b_idx >= 0)
		{
			size_a = (a_idx >= 0) ? a_shape[a_idx] : 1;
			size_b = (b_idx >= 0) ? b_shape[b_idx] : 1;

			// Broadcasting rules check
			if (size_a == size_b)
				{
					target_shape->s[out_idx] = size_a;
					if (a_idx >= 0) out_stride_a->s[out_idx] = a_strides[a_idx];
					if (b_idx >= 0) out_stride_b->s[out_idx] = b_strides[b_idx];
				}
			// B dictates size, A broadcasts
			else if (size_a == 1)
				{
					target_shape->s[out_idx] = size_b;
					out_stride_a->s[out_idx] = 0; // Forced to 0 because size_a is 1
					if (b_idx >= 0) out_stride_b->s[out_idx] = b_strides[b_idx];
				}
			// A dictates size, B broadcasts
			else if (size_b == 1)
				{
					target_shape->s[out_idx] = size_a;
					if (a_idx >= 0) out_stride_a->s[out_idx] = a_strides[a_idx];
					out_stride_b->s[out_idx] = 0; // Forced to 0 because size_b is 1
				}
			else
				{
					fprintf (stderr, "Error: Incompatible shapes for broadcasting!\n");
					abort ();
				}

			out_idx--;
			a_idx--;
			b_idx--;
		}
}

static int
bcl_tensor_shapes_are_equal (const BclTensor *a, const BclTensor *b)
{
	const size_t *a_shape = NULL;
	const size_t *b_shape = NULL;

	if (bcl_tensor_rank (a) != bcl_tensor_rank (b))
		return 0;

	a_shape = bcl_tensor_shape (a);
	b_shape = bcl_tensor_shape (b);

	for (size_t i = 0; i < bcl_tensor_rank (a); i++)
		if (a_shape[i] != b_shape[i])
			return 0;

	return 1;
}

BclError
bcl_tensor_add (const BclEnv *env, const BclTensor *a, const BclTensor *b, BclTensor *c)
{
	assert (env != NULL);
	assert (a != NULL && b != NULL && c != NULL);

	cl_int cl_err = CL_SUCCESS;

	cl_kernel kernel = NULL;
	cl_mem data_a = NULL;
	cl_mem data_b = NULL;
	cl_mem data_c = NULL;

	size_t global_work_size = 0;
	cl_uint n = 0;

	data_a = bcl_tensor_data (a);
	data_b = bcl_tensor_data (b);
	data_c = bcl_tensor_data (c);

	if (bcl_tensor_shapes_are_equal (a, b)
			&& bcl_tensor_is_contiguous (a)
			&& bcl_tensor_is_contiguous (b))
		{
			kernel = bcl_env_kernel (env, BCL_KERNEL_ADD);

			global_work_size = bcl_tensor_size (a);
			n = bcl_tensor_size (a);

			clSetKernelArg (kernel, 0, sizeof (cl_mem), &data_a);
			clSetKernelArg (kernel, 1, sizeof (cl_mem), &data_b);
			clSetKernelArg (kernel, 2, sizeof (cl_mem), &data_c);
			clSetKernelArg (kernel, 3, sizeof (cl_uint), &n);
		}
	else
		{
			kernel = bcl_env_kernel (env, BCL_KERNEL_ADD_BROADCAST);

			cl_uint8 target_shape = {{0}};
			cl_uint8 stride_a = {{0}};
			cl_uint8 stride_b = {{0}};

			compute_broadcasting_metadata (a, b, &target_shape, &stride_a, &stride_b);

			global_work_size = bcl_tensor_size (c);
			n = bcl_tensor_size (c);

			clSetKernelArg (kernel, 0, sizeof (cl_mem), &data_a);
			clSetKernelArg (kernel, 1, sizeof (cl_mem), &data_b);
			clSetKernelArg (kernel, 2, sizeof (cl_mem), &data_c);
			clSetKernelArg (kernel, 3, sizeof (cl_uint8), &target_shape);
			clSetKernelArg (kernel, 4, sizeof (cl_uint8), &stride_a);
			clSetKernelArg (kernel, 5, sizeof (cl_uint8), &stride_b);
			clSetKernelArg (kernel, 6, sizeof (cl_uint), &n);
		}

	cl_err = clEnqueueNDRangeKernel (bcl_env_queue (env), kernel,
			1, NULL, &global_work_size, NULL, 0, NULL, NULL);

	return cl_err != CL_SUCCESS ? cl_err : BCL_SUCCESS;
}

BclError
bcl_tensor_add_scalar (const BclEnv *env, const BclTensor *a, float s, BclTensor *c)
{
	assert (env != NULL);
	assert (a != NULL && c != NULL);

	cl_int cl_err = CL_SUCCESS;
	cl_kernel kernel = NULL;
	cl_mem data_a = NULL;
	cl_mem data_c = NULL;
	size_t global_work_size = 0;
	cl_uint n = 0;

	kernel = bcl_env_kernel (env, BCL_KERNEL_ADD_SCALAR);
	data_a = bcl_tensor_data (a);
	data_c = bcl_tensor_data (c);
	global_work_size = bcl_tensor_size (a);
	n = global_work_size;

	clSetKernelArg (kernel, 0, sizeof (cl_mem), &data_a);
	clSetKernelArg (kernel, 1, sizeof (float), &s);
	clSetKernelArg (kernel, 2, sizeof (cl_mem), &data_c);
	clSetKernelArg (kernel, 3, sizeof (cl_uint), &n);

	cl_err = clEnqueueNDRangeKernel (bcl_env_queue (env), kernel,
		1, NULL, &global_work_size, NULL, 0, NULL, NULL);

	return cl_err != CL_SUCCESS ? cl_err : BCL_SUCCESS;
}
