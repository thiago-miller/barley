#include <check.h>
#include <bcl/env.h>
#include <bcl/tensor.h>
#include <bcl/tensor_ops.h>
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

	c = bcl_tensor_add (env, a, b, c, &err);
	ck_assert_ptr_nonnull (c);
	ck_assert_int_eq (err, BCL_SUCCESS);

	data_c = bcl_tensor_read (env, c, 0, 0, &err);
	ck_assert_int_eq (err, BCL_SUCCESS);
	ck_assert_ptr_nonnull (data_c);

	for (size_t i = 0; i < shape; i++)
		{
			float expected = host_a[i] + host_b[i];
			ck_assert_float_eq_tol (data_c[i], expected, 1e-5);
		}

	free (data_c);
	bcl_tensor_free (a);
	bcl_tensor_free (b);
	bcl_tensor_free (c);
}
END_TEST

START_TEST (test_tensor_add_scalar)
{
	BclError err = BCL_SUCCESS;
	BclTensor *a = NULL;
	BclTensor *c = NULL;

	float *data_c = NULL;
	float s = 0;

	size_t shape[] = {5, 3};

	float data_a[] = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		4.0f, 5.0f, 6.0f
	};

	s = 10.0f;

	float expected[] = {
		11.0f, 12.0f, 13.0f,
		14.0f, 15.0f, 16.0f,
		11.0f, 12.0f, 13.0f,
		14.0f, 15.0f, 16.0f,
		14.0f, 15.0f, 16.0f
	};

	a = bcl_tensor_alloc (env, 2, shape, data_a, 1, &err);
	ck_assert_int_eq (err, BCL_SUCCESS);

	c = bcl_tensor_alloc (env, 2, shape, NULL, 0, &err);
	ck_assert_int_eq (err, BCL_SUCCESS);

	c = bcl_tensor_add_scalar (env, a, s, c, &err);
	ck_assert_ptr_nonnull (c);
	ck_assert_int_eq (err, BCL_SUCCESS);

	data_c = bcl_tensor_read (env, c, 0, 0, &err);

	for (int i = 0; i < 15; i++)
		ck_assert_float_eq_tol (data_c[i], expected[i], 1e-5);
}
END_TEST

START_TEST (test_tensor_add_broadcast_2d_1d)
{
	BclError err = BCL_SUCCESS;
	BclTensor *a = NULL;
	BclTensor *b = NULL;
	BclTensor *c = NULL;
	float *data_c = NULL;

	size_t shape_a[] = {2, 3};
	float data_a[] = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f
	};

	size_t shape_b[] = {3};
	float data_b[] = {
		10.0f, 20.0f, 30.0f
	};

	float expected[6] = {
		11.0f, 22.0f, 33.0f,
		14.0f, 25.0f, 36.0f
	};

	a = bcl_tensor_alloc (env, 2, shape_a, data_a, 0, &err);
	ck_assert_int_eq (err, BCL_SUCCESS);

	b = bcl_tensor_alloc (env, 1, shape_b, data_b, 0, &err);
	ck_assert_int_eq (err, BCL_SUCCESS);

	c = bcl_tensor_add (env, a, b, NULL, &err);
	ck_assert_ptr_nonnull (c);
	ck_assert_int_eq (err, BCL_SUCCESS);

	data_c = bcl_tensor_read (env, c, 0, 0, &err);

	for (int i = 0; i < 6; i++)
		ck_assert_float_eq_tol (data_c[i], expected[i], 1e-5);

	free (data_c);
	bcl_tensor_free (a);
	bcl_tensor_free (b);
	bcl_tensor_free (c);
}
END_TEST

START_TEST (test_tensor_sum_axis1)
{
	BclError err = BCL_SUCCESS;
	BclTensor *a = NULL;
	BclTensor *c = NULL;
	float *data_c = NULL;

	size_t shape_a[2] = { 4, 4 };
	float data_a[16] = {
		1.0f, 2.0f, 3.0f, 4.0f,
		4.0f, 5.0f, 6.0f, 7.0f,
		1.0f, 2.0f, 3.0f, 4.0f,
		4.0f, 5.0f, 6.0f, 7.0f,
	};

	float expected[4] = { 10.0f, 22.0f, 10.0f, 22.0f };

	a = bcl_tensor_alloc (env, 2, shape_a, data_a, 0, &err);
	ck_assert_int_eq (err, BCL_SUCCESS);

	c = bcl_tensor_sum (env, a, 1, NULL, &err);
	ck_assert_ptr_nonnull (c);
	ck_assert_int_eq (err, BCL_SUCCESS);
	ck_assert_int_eq (bcl_tensor_rank (c), 1);

	data_c = bcl_tensor_read (env, c, 0, 0, &err);
	ck_assert_int_eq (err, BCL_SUCCESS);
	ck_assert_ptr_nonnull (data_c);

	for (int i = 0; i < 4; i++)
		ck_assert_float_eq_tol (data_c[i], expected[i], 1e-5);

	free (data_c);
	bcl_tensor_free (a);
	bcl_tensor_free (c);
}
END_TEST

START_TEST (test_tensor_sum_axis0)
{
	BclError err = BCL_SUCCESS;
	BclTensor *a = NULL;
	BclTensor *c = NULL;
	float *data_c = NULL;

	size_t shape_a[2] = { 2, 3 };
	float data_a[6] = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f
	};

	float expected[3] = { 5.0f, 7.0f, 9.0f }; /* 1+4, 2+5, 3+6 */

	a = bcl_tensor_alloc (env, 2, shape_a, data_a, 0, &err);
	ck_assert_int_eq (err, BCL_SUCCESS);

	c = bcl_tensor_sum (env, a, 0, NULL, &err);
	ck_assert_ptr_nonnull (c);
	ck_assert_int_eq (err, BCL_SUCCESS);
	ck_assert_int_eq (bcl_tensor_rank (c), 1);
	ck_assert_int_eq (bcl_tensor_shape (c)[0], 3);

	data_c = bcl_tensor_read (env, c, 0, 0, &err);
	ck_assert_int_eq (err, BCL_SUCCESS);
	ck_assert_ptr_nonnull (data_c);

	for (int i = 0; i < 3; i++)
		ck_assert_float_eq_tol (data_c[i], expected[i], 1e-5);

	free (data_c);
	bcl_tensor_free (a);
	bcl_tensor_free (c);
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
	tcase_add_test (tc_core, test_tensor_add_scalar);
	tcase_add_test (tc_core, test_tensor_add_broadcast_2d_1d);
	tcase_add_test (tc_core, test_tensor_sum_axis1);
	tcase_add_test (tc_core, test_tensor_sum_axis0);

	suite_add_tcase (s, tc_core);

	return s;
}
