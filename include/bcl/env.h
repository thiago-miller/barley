#pragma once

#include <bcl/cl.h>
#include <bcl/error.h>

#define BCL_KERNEL_COUNT 1

typedef struct _BclEnv BclEnv;
typedef enum _BclKernelID _BclKernelID;

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
	BCL_KERNEL_ADD = 0
};

BclEnv * bcl_env_alloc (cl_device_type device_type, BclError *out_err);
void     bcl_env_free  (BclEnv *env);
