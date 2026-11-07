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
	uint gid = get_global_id (0);
	if (gid >= n)
		return;

	uint idx_a = 0;
	uint idx_b = 0;

	// Calculte BclTensor's a and b indexes
	__compute_broadcast_index (&idx_a, &idx_b, target_shape, stride_a, stride_b, gid);

	c[gid] = a[idx_a] + b[idx_b];
}

static inline uint
__compute_reduction_base_index (
	const uint8 shape,
	const uint8 stride,
	const uint  axis,
	uint gid
)
{
	uint base_idx = 0;
	uint8 coords = (uint8)(0);

	// Unpack 'gid' into 8D output coordinates.
	for (int i = 7; i >= 0; i--)
		{
			if (i == (int) axis)
				continue;

			uint dim_size = shape[i];

			coords[i] = gid % dim_size;
			gid /= dim_size;
		}

	// Map 8D coordinates to flat input memory index
	for (int i = 0; i < 8; i++)
		if (i != (int) axis)
			base_idx += coords[i] * stride[i];

	return base_idx;
}

__kernel void
tensor_sum (
	__global const float *a,
	__global       float *c,
	const uint8 shape_a,
	const uint8 stride_a,
	const uint  axis,
	const uint  n
)
{
	uint gid = get_global_id (0);
	if (gid >= n)
		return;

	uint reduced_shape = 0;
	uint reduced_stride = 0;
	uint base_idx = 0;
	float sum = 0.0f;

	// Compute base index
	base_idx = __compute_reduction_base_index (shape_a, stride_a, axis, gid);

	// Perform reduction along the target axis
	reduced_shape = shape_a[axis];
	reduced_stride = stride_a[axis];

	for (int i = 0; i < (int) reduced_shape; i++)
		sum += a[base_idx + (i * reduced_stride)];

	c[gid] = sum;
}
