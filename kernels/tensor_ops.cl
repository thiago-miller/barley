__kernel void
tensor_add (
	__global const float *a,
	__global const float *b,
	__global       float *c,
	const uint n
)
{
	uint gid = get_global_id (0);
	if (gid < n)
		c[gid] = a[gid] + b[gid];
}

__kernel void
tensor_add_scalar (
	__global const float *a,
	const float s,
	__global       float *c,
	const uint  n
)
{
	uint gid = get_global_id (0);
	if (gid < n)
		c[gid] = a[gid] + s;
}

static inline void
__compute_broadcast_index (
	uint *idx_a,
	uint *idx_b,
	const uint8 target_shape,
	const uint8 stride_a,
	const uint8 stride_b,
	uint gid
)
{
	uint cur_idx_a = 0;
	uint cur_idx_b = 0;

	// Decode the linear 'gid' uinto coordinates and compute source offsets
	for (int i = 7; i >= 0; i--)
		{
			uint coord = gid % target_shape[i];
			gid /= target_shape[i];

			cur_idx_a += coord * stride_a[i];
			cur_idx_b += coord * stride_b[i];
		}

	*idx_a = cur_idx_a;
	*idx_b = cur_idx_b;
}

__kernel void
tensor_add_broadcast (
	__global const float* a,
	__global const float* b,
	__global       float* c,
	const uint8 target_shape,
	const uint8 stride_a,
	const uint8 stride_b,
	const uint  n
)
{
	// Get the flat, linear index for this GPU thread
	uint gid = get_global_id (0);
	if (gid >= n)
		return;

	uint idx_a = 0;
	uint idx_b = 0;

	// Calculte BclTensor's a and b indexes
	__compute_broadcast_index (&idx_a, &idx_b, target_shape, stride_a, stride_b, gid);

	// Write directly to the output. No output stride math needed!
	c[gid] = a[idx_a] + b[idx_b];
}
