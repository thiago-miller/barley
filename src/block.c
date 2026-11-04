#include <bcl/block.h>
#include <assert.h>

BclBlock *
bcl_block_alloc (const BclEnv *env, const float *data, size_t size, int ro, BclError *out_err)
{
	assert (env != NULL);

	BclBlock *block = NULL;

	BclError err = BCL_SUCCESS;
	cl_int cl_err = CL_SUCCESS;

	cl_mem_flags flags;
	cl_mem d_data = NULL;

	flags = ro ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;

	if (data != NULL)
		flags |= CL_MEM_COPY_HOST_PTR;

	d_data = clCreateBuffer (bcl_env_context (env),
			flags, sizeof (float) * size, (float *) data, &cl_err);
	if (cl_err != CL_SUCCESS)
		{
			err = cl_err;
			goto ERR;
		}

	block = malloc (sizeof (BclBlock));
	if (block == NULL)
		{
			err = errno;
			goto ERR;
		}

	*block = (BclBlock) {
		.flags = flags,
		.data  = d_data,
		.size  = size
	};

RET:
	if (out_err != NULL)
		*out_err = err;

	return block;

ERR:
	if (d_data != NULL)
		clReleaseMemObject (d_data);

	if (block != NULL)
		free (block);

	goto RET;
}

void
bcl_block_free (BclBlock *block)
{
	if (block == NULL)
		return;

	clReleaseMemObject (block->data);
	free (block);
}

float *
bcl_block_read (const BclBlock *block, const BclEnv *env, size_t offset, size_t size, BclError *out_err)
{
	assert (block != NULL);
	assert (env != NULL);

	BclError err = BCL_SUCCESS;
	cl_int cl_err = CL_SUCCESS;

	float *data = NULL;

	if (size == 0)
		size = bcl_block_size (block);

	data = malloc (sizeof (float) * size);
	if (data == NULL)
		{
			err = errno;
			goto ERR;
		}

	cl_err = clEnqueueReadBuffer (bcl_env_queue (env), bcl_block_data (block),
			CL_TRUE, offset, sizeof (float) * size, data, 0, NULL, NULL);
	if (cl_err != CL_SUCCESS)
		{
			err = cl_err;
			goto ERR;
		}

RET:
	if (out_err != NULL)
		*out_err = err;

	return data;

ERR:
	if (data != NULL)
		free (data);

	goto RET;
}

BclError
bcl_block_write (BclBlock *block, const BclEnv *env, const float *data, size_t offset, size_t size)
{
	assert (block != NULL);
	assert (env != NULL);
	assert (data != NULL);

	BclError err = BCL_SUCCESS;
	cl_int cl_err = CL_SUCCESS;

	cl_err = clEnqueueWriteBuffer (bcl_env_queue (env), bcl_block_data (block),
			CL_TRUE, offset, sizeof (float) * size, data, 0, NULL, NULL);
	if (cl_err != CL_SUCCESS)
		err = cl_err;

	return err;
}
