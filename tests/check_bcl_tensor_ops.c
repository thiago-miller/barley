#include <check.h>
#include <bcl/env.h>
#include <bcl/tensor.h>
#include <bcl/tensor_ops.h>
#include <bcl/block.h>
#include <math.h>

static BclEnv *env = NULL;

static void
setup (void)
{
	BclError err = BCL_SUCCESS;

	env = bcl_env_alloc (CL_DEVICE_TYPE_GPU, &err);
	if (err != BCL_SUCCESS)
		env = bcl_env_alloc (CL_DEVICE_TYPE_CPU, &err);

	ck_assert_msg (err == BCL_SUCCESS,
			"Failed to init OpenCL env");
}

static void
teardown (void)
{
	if (env != NULL)
		{
			bcl_env_free (env);
			env = NULL;
		}
}

START_TEST (test_tensor_add_basic)
{
	BclError err = BCL_SUCCESS;
	BclTensor *a = NULL;
	BclTensor *b = NULL;
	BclTensor *c = NULL;
	float *data_c = NULL;

	size_t shape = 4;
	float host_a[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
	float host_b[4] = { 10.0f, 20.0f, 30.0f, 40.0f };

	a = bcl_tensor_alloc (env, 1, &shape, host_a, 1, &err);
	ck_assert_int_eq (err, BCL_SUCCESS);

	b = bcl_tensor_alloc (env, 1, &shape, host_b, 1, &err);
	ck_assert_int_eq (err, BCL_SUCCESS);

	c = bcl_tensor_alloc (env, 1, &shape, NULL, 0, &err);
	ck_assert_int_eq (err, BCL_SUCCESS);

	err = bcl_tensor_add (env, a, b, c);
	ck_assert_int_eq (err, BCL_SUCCESS);

	data_c = bcl_block_read (c->block, env, 0, 0, &err);
	ck_assert_int_eq (err, BCL_SUCCESS);
	ck_assert_ptr_nonnull (data_c);

	for (size_t i = 0; i < shape; i++)
		{
			float expected = host_a[i] + host_b[i];
			ck_assert_msg (fabsf (data_c[i] - expected) < 1e-5f,
					"Error at index %zu: expected %.1f, got %.1f",
					i, expected, data_c[i]);
		}

	if (data_c) free (data_c);
	if (a) bcl_tensor_free (a);
	if (b) bcl_tensor_free (b);
	if (c) bcl_tensor_free (c);
}
END_TEST

Suite *
make_tensor_ops_suite (void)
{
	Suite *s;
	TCase *tc_core;

	s = suite_create ("TensorOps");
	tc_core  = tcase_create ("Core");

	tcase_add_checked_fixture (tc_core, setup, teardown);

	tcase_add_test (tc_core, test_tensor_add_basic);

	suite_add_tcase (s, tc_core);

	return s;
}
