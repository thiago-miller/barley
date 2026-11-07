#pragma once

#include <bcl/cl.h>
#include <bcl/error.h>

#define BCL_KERNEL_COUNT 4

typedef struct _BclEnv BclEnv;
typedef enum _BclKernelID BclKernelID;

struct _BclEnv
{
	cl_device_type   device_type;
	cl_device_id     device_id;
	cl_context       context;
	cl_command_queue queue;
	cl_program       program;
	cl_kernel        kernels[BCL_KERNEL_COUNT];
};

enum _BclKernelID
{
	BCL_KERNEL_ADD = 0,
	BCL_KERNEL_ADD_SCALAR,
	BCL_KERNEL_ADD_BROADCAST,
	BCL_KERNEL_SUM
};

static inline cl_device_type
bcl_env_device_type (const BclEnv *env)
{
	return env != NULL ? env->device_type : 0;
}

static inline cl_device_id
bcl_env_device_id (const BclEnv *env)
{
	return env != NULL ? env->device_id : NULL;
}

static inline cl_context
bcl_env_context (const BclEnv *env)
{
	return env != NULL ? env->context : NULL;
}

static inline cl_command_queue
bcl_env_queue (const BclEnv *env)
{
	return env != NULL ? env->queue : NULL;
}

static inline cl_program
bcl_env_program (const BclEnv *env)
{
	return env != NULL ? env->program : NULL;
}

static inline cl_kernel
bcl_env_kernel (const BclEnv *env, size_t i)
{
	return env != NULL ? env->kernels[i] : NULL;
}

BclEnv * bcl_env_alloc (cl_device_type device_type, BclError *out_err);
void     bcl_env_free  (BclEnv *env);
