__kernel void
tensor_add (int n, __global const float *a, __global const float *b, __global float *c)
{
	int i = get_global_id (0);
	if (i < n)
		c[i] = a[i] + b[i];
}
