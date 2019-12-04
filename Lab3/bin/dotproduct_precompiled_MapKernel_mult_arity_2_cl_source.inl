
class CLWrapperClass_dotproduct_precompiled_MapKernel_mult_arity_2
{
public:
	
	static cl_kernel kernels(size_t deviceID, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID];
	}
	
	static void initialize()
	{
		static bool initialized = false;
		if (initialized)
			return;
		
		std::string source = skepu2::backend::cl_helpers::replaceSizeT(R"###(
#define SKEPU_USING_BACKEND_CL 1

typedef struct{
	size_t i;
} index1_t;

typedef struct {
	size_t row;
	size_t col;
} index2_t;

size_t get_device_id()
{
	return SKEPU_INTERNAL_DEVICE_ID;
}

#define VARIANT_OPENCL(block) block
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block)

static float mult(float a, float b)
{
	return a*b;
}


__kernel void dotproduct_precompiled_MapKernel_mult_arity_2(__global float *a, __global float *b,  __global float* output, size_t w, size_t n, size_t base)
{
	size_t i = get_global_id(0);
	size_t gridSize = get_local_size(0) * get_num_groups(0);
	
	
	while (i < n)
	{
		
		output[i] = mult(a[i], b[i]);
		i += gridSize;
	}
}
)###");
		
		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu2::backend::Device_CL *device : skepu2::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu2::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel = clCreateKernel(program, "dotproduct_precompiled_MapKernel_mult_arity_2", &err);
			CL_CHECK_ERROR(err, "Error creating map kernel '" << "dotproduct_precompiled_MapKernel_mult_arity_2" << "'");
			
			kernels(counter++, &kernel);
		}
		
		initialized = true;
	}
	
	static void map
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		skepu2::backend::DeviceMemPointer_CL<float> *a, skepu2::backend::DeviceMemPointer_CL<float> *b,  skepu2::backend::DeviceMemPointer_CL<float> *output,
		size_t w, size_t n, size_t base
	)
	{
		skepu2::backend::cl_helpers::setKernelArgs(kernels(deviceID), a->getDeviceDataPointer(), b->getDeviceDataPointer(),  output->getDeviceDataPointer(), w, n, base);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernels(deviceID), 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Map kernel");
	}
};
