#include <bcl/tensor_ops.h>
#include <limits.h>
#include <string.h>

#define KARG(x) { sizeof (x), &(x) }

typedef struct _KernelArg     KernelArg;
typedef struct _BroadcastPlan BroadcastPlan;
typedef struct _ReductionPlan ReductionPlan;

struct _KernelArg
{
	size_t      size;
	const void *value;
};

struct _BroadcastPlan
{
	size_t   rank;
	size_t   shape[BCL_TENSOR_MAX_RANK];
	cl_uint8 kshape;
	cl_uint8 kstride_a;
	cl_uint8 kstride_b;
};

struct _ReductionPlan
{
	size_t   out_rank;
	size_t   out_shape[BCL_TENSOR_MAX_RANK];
	size_t   out_size;
	cl_uint8 kshape;
	cl_uint8 kstride;
	cl_uint  kaxis;
};

static BclError
launch_1d (const BclEnv *env, BclKernelID id,
		const KernelArg *args, cl_uint nargs, size_t global_size)
{
	cl_kernel kernel = bcl_env_kernel (env, id);

	for (cl_uint i = 0; i < nargs; i++)
		{
			cl_int e = clSetKernelArg (kernel, i, args[i].size, args[i].value);
			if (e != CL_SUCCESS)
				return e;
		}

	return clEnqueueNDRangeKernel (bcl_env_queue (env),
			kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
}

/* The kernels use 32-bit indices. */
static int
fits_in_uint (size_t n)
{
	return n <= UINT_MAX;
}

static int
same_shape (const BclTensor *a, const BclTensor *b)
{
	if (bcl_tensor_rank (a) != bcl_tensor_rank (b))
		return 0;

	return memcmp (bcl_tensor_shape (a), bcl_tensor_shape (b),
			bcl_tensor_rank (a) * sizeof (size_t)) == 0;
}

/* Result of broadcasting two tensors, laid out the way the kernel wants:
 * right-aligned, padded to BCL_TENSOR_MAX_RANK with shape=1, stride=0. */
static BclError
broadcast_plan_init (BroadcastPlan *plan, const BclTensor *a, const BclTensor *b)
{
	const size_t ra = bcl_tensor_rank (a);
	const size_t rb = bcl_tensor_rank (b);
	const size_t *sha = bcl_tensor_shape (a);
	const size_t *shb = bcl_tensor_shape (b);
	const size_t *sta = bcl_tensor_strides (a);
	const size_t *stb = bcl_tensor_strides (b);

	plan->rank = ra > rb ? ra : rb;
	if (plan->rank > BCL_TENSOR_MAX_RANK)
		return BCL_ERROR_INVALID_ARGUMENT;

	for (size_t i = 0; i < BCL_TENSOR_MAX_RANK; i++)
		{
			plan->kshape.s[i] = 1;
			plan->kstride_a.s[i] = 0;
			plan->kstride_b.s[i] = 0;
		}

	/* k = 0 is the trailing dimension. */
	for (size_t k = 0; k < plan->rank; k++)
		{
			const size_t da = k < ra ? sha[ra - 1 - k] : 1;
			const size_t db = k < rb ? shb[rb - 1 - k] : 1;
			const size_t out = BCL_TENSOR_MAX_RANK - 1 - k;

			if (da != db && da != 1 && db != 1)
				return BCL_ERROR_SHAPE_MISMATCH;

			const size_t d = (da == 1) ? db : da;
			if (!fits_in_uint (d))
				return BCL_ERROR_INVALID_ARGUMENT;

			plan->kshape.s[out] = (cl_uint) d;
			/* A broadcast (size-1 or missing) dimension has stride 0. */
			if (k < ra && da != 1) plan->kstride_a.s[out] = (cl_uint) sta[ra - 1 - k];
			if (k < rb && db != 1) plan->kstride_b.s[out] = (cl_uint) stb[rb - 1 - k];

			plan->shape[plan->rank - 1 - k] = d;
		}

	return BCL_SUCCESS;
}

/* Returns the tensor to write into: c itself (validated), or a fresh one.
 * *allocated tells the caller whether it must free the result on failure. */
static BclTensor *
output_prepare (const BclEnv *env, BclTensor *c,
		size_t rank, const size_t *shape,
		int *allocated, BclError *err)
{
	*allocated = 0;
	*err = BCL_SUCCESS;

	if (c == NULL)
		{
			c = bcl_tensor_alloc (env, rank, shape, NULL, 0, err);
			if (*err != BCL_SUCCESS)
				return NULL;
			*allocated = 1;
			return c;
		}

	if (bcl_tensor_rank (c) != rank
			|| memcmp (bcl_tensor_shape (c), shape, rank * sizeof (size_t)) != 0)
		{
			*err = BCL_ERROR_SHAPE_MISMATCH;
			return NULL;
		}

	/* The kernels write c[gid] directly, so c must be contiguous
	* and writable. */
	if (!bcl_tensor_is_contiguous (c)
			|| (bcl_block_flags (c->block) & CL_MEM_READ_ONLY))
		{
			*err = BCL_ERROR_INVALID_ARGUMENT;
			return NULL;
		}

	return c;
}

static BclTensor *
output_finish (BclTensor *out, int allocated, BclError err, BclError *out_err)
{
	if (err != BCL_SUCCESS)
		{
			if (allocated)
				bcl_tensor_free (out);
			out = NULL;
		}

	if (out_err != NULL)
		*out_err = err;

	return out;
}

/* ------------------------------------------------------------------ */
/* add                                                                */
/* ------------------------------------------------------------------ */

static BclError
add_launch (const BclEnv *env, const BclTensor *a, const BclTensor *b,
		BclTensor *c, const BroadcastPlan *plan)
{
	cl_mem data_a = bcl_tensor_data (a);
	cl_mem data_b = bcl_tensor_data (b);
	cl_mem data_c = bcl_tensor_data (c);

	if (!fits_in_uint (bcl_tensor_size (c)))
		return BCL_ERROR_INVALID_ARGUMENT;
	cl_uint n = (cl_uint) bcl_tensor_size (c);

	/* Fast path: same shape, both dense. */
	if (same_shape (a, b)
			&& bcl_tensor_is_contiguous (a)
			&& bcl_tensor_is_contiguous (b))
		{
			KernelArg args[] = { KARG (data_a), KARG (data_b),
					KARG (data_c), KARG (n) };
			return launch_1d (env, BCL_KERNEL_ADD, args, 4, n);
		}

	KernelArg args[] = {
		KARG (data_a), KARG (data_b), KARG (data_c),
		KARG (plan->kshape), KARG (plan->kstride_a), KARG (plan->kstride_b),
		KARG (n)
	};
	return launch_1d (env, BCL_KERNEL_ADD_BROADCAST, args, 7, n);
}

BclTensor *
bcl_tensor_add (const BclEnv *env, const BclTensor *a, const BclTensor *b,
		BclTensor *c, BclError *out_err)
{
	BclError err = BCL_SUCCESS;
	BroadcastPlan plan;
	BclTensor *out = NULL;
	int allocated = 0;

	if (env == NULL || a == NULL || b == NULL)
		err = BCL_ERROR_INVALID_ARGUMENT;
	else
		err = broadcast_plan_init (&plan, a, b);

	if (err == BCL_SUCCESS)
		out = output_prepare (env, c, plan.rank, plan.shape, &allocated, &err);

	if (err == BCL_SUCCESS)
		err = add_launch (env, a, b, out, &plan);

	return output_finish (out, allocated, err, out_err);
}

BclTensor *
bcl_tensor_add_scalar (const BclEnv *env, const BclTensor *a, float s,
		BclTensor *c, BclError *out_err)
{
	BclError err = BCL_SUCCESS;
	BclTensor *out = NULL;
	int allocated = 0;

	if (env == NULL || a == NULL)
		err = BCL_ERROR_INVALID_ARGUMENT;
	else
		out = output_prepare (env, c, bcl_tensor_rank (a), bcl_tensor_shape (a),
				&allocated, &err);

	if (err == BCL_SUCCESS && !fits_in_uint (bcl_tensor_size (a)))
		err = BCL_ERROR_INVALID_ARGUMENT;

	if (err == BCL_SUCCESS)
		{
			cl_mem data_a = bcl_tensor_data (a);
			cl_mem data_c = bcl_tensor_data (out);
			cl_uint n = (cl_uint) bcl_tensor_size (a);
			KernelArg args[] = { KARG (data_a), KARG (s), KARG (data_c), KARG (n) };

			err = launch_1d (env, BCL_KERNEL_ADD_SCALAR, args, 4, n);
		}

	return output_finish (out, allocated, err, out_err);
}

/* ------------------------------------------------------------------ */
/* sum                                                                */
/* ------------------------------------------------------------------ */

static BclError
reduction_plan_init (ReductionPlan *plan, const BclTensor *a, size_t axis)
{
	const size_t rank = bcl_tensor_rank (a);
	const size_t *shape = bcl_tensor_shape (a);
	const size_t *strides = bcl_tensor_strides (a);
	size_t pad = 0;

	if (rank == 0 || rank > BCL_TENSOR_MAX_RANK || axis >= rank)
		return BCL_ERROR_INVALID_ARGUMENT;

	/* The kernel indexes input elements with 32-bit arithmetic. */
	if (!fits_in_uint (bcl_tensor_size (a)))
		return BCL_ERROR_SIZE_OVERFLOW;

	pad = BCL_TENSOR_MAX_RANK - rank;

	for (size_t i = 0; i < BCL_TENSOR_MAX_RANK; i++)
		{
			plan->kshape.s[i] = 1;
			plan->kstride.s[i] = 0;
		}

	plan->out_rank = 0;
	plan->out_size = 1;

	for (size_t i = 0; i < rank; i++)
		{
			/* A zero dimension would make the kernel divide by zero. */
			if (shape[i] == 0)
				return BCL_ERROR_INVALID_ARGUMENT;
			if (!fits_in_uint (shape[i]) || !fits_in_uint (strides[i]))
				return BCL_ERROR_SIZE_OVERFLOW;

			plan->kshape.s[pad + i] = (cl_uint) shape[i];
			plan->kstride.s[pad + i] = (cl_uint) strides[i];

			if (i != axis)
				{
					plan->out_shape[plan->out_rank++] = shape[i];
					plan->out_size *= shape[i];
				}
		}

	plan->kaxis = (cl_uint) (pad + axis);

	return BCL_SUCCESS;
}

static BclError
sum_launch (const BclEnv *env, const BclTensor *a, BclTensor *c,
		const ReductionPlan *plan)
{
	cl_mem data_a = bcl_tensor_data (a);
	cl_mem data_c = bcl_tensor_data (c);
	cl_uint n = (cl_uint) plan->out_size;

	/* Order must match the kernel signature. */
	KernelArg args[] = {
		KARG (data_a), KARG (data_c),
		KARG (plan->kshape), KARG (plan->kstride),
		KARG (plan->kaxis), KARG (n)
	};

	return launch_1d (env, BCL_KERNEL_SUM, args, 6, n);
}

BclTensor *
bcl_tensor_sum (const BclEnv *env, const BclTensor *a, size_t axis, BclTensor *c, BclError *out_err)
{
	BclError err = BCL_SUCCESS;
	ReductionPlan plan;
	BclTensor *out = NULL;
	int allocated = 0;

	if (env == NULL || a == NULL)
		err = BCL_ERROR_INVALID_ARGUMENT;
	else
		err = reduction_plan_init (&plan, a, axis);

	if (err == BCL_SUCCESS)
		out = output_prepare (env, c, plan.out_rank, plan.out_shape,
				&allocated, &err);

	/* Each output element reads many inputs: in-place is not allowed. */
	if (err == BCL_SUCCESS && bcl_tensor_data (out) == bcl_tensor_data (a))
		err = BCL_ERROR_ALIASING;

	if (err == BCL_SUCCESS)
		err = sum_launch (env, a, out, &plan);

	return output_finish (out, allocated, err, out_err);
}
