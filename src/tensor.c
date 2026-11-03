#include <bcl/tensor.h>

#include <string.h>
#include <errno.h>
#include <assert.h>

BclTensor *
bcl_tensor_alloc (cl_context context, size_t rank, size_t *shape, float *data, int ro, BclError *out_err)
{
	assert (rank > 0);
	assert (shape != NULL);

	BclTensor *tensor = NULL;
	BclBlock *block = NULL;
	BclError err = BCL_SUCCESS;

	size_t *shape_copy = NULL;
	size_t *strides = NULL;
	size_t size = 1;

	shape_copy = malloc (sizeof (size_t) * rank);
	if (shape_copy == NULL)
		{
			err = errno;
			goto ERR;
		}

	memcpy (shape_copy, shape, sizeof (size_t) * rank);

	strides = malloc (sizeof (size_t) * rank);
	if (strides == NULL)
		{
			err = errno;
			goto ERR;
		}

	for (size_t i = 0; i < rank; i++)
		{
			strides[i] = 1;
			for (size_t j = i + 1; j < rank; j++)
				strides[i] *= shape[j];
		}

	for (size_t i = 0; i < rank; i++)
		size *= shape[i];

	block = bcl_block_alloc (context, data, size, ro, &err);
	if (err != BCL_SUCCESS)
		goto ERR;

	tensor = malloc (sizeof (BclTensor));
	if (tensor == NULL)
		{
			err = errno;
			goto ERR;
		}

	*tensor = (BclTensor) {
		.rank    = rank,
		.shape   = shape_copy,
		.strides = strides,
		.offset  = 0,
		.size    = size,
		.owner   = 1,
		.block   = block
	};

RET:
	if (out_err != NULL)
		*out_err = err;

	return tensor;

ERR:
	if (shape_copy != NULL)
		free (shape_copy);

	if (strides != NULL)
		free (strides);

	if (block != NULL)
		bcl_block_free (block);

	if (tensor != NULL)
		free (tensor);

	goto RET;
}

void
bcl_tensor_free (BclTensor *tensor)
{
	if (tensor == NULL)
		return;

	if (tensor->owner)
		bcl_block_free (tensor->block);

	free (tensor->shape);
	free (tensor->strides);
	free (tensor);
}
