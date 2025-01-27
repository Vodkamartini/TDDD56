
class CLWrapperClass_dotproduct_precompiled_ReduceKernel_scalarAddFunc
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

static float scalarAddFunc(float a, float b)
{
	return a+b;
}


__kernel void dotproduct_precompiled_ReduceKernel_scalarAddFunc(__global float* input, __global float* output, size_t n, __local float* sdata)
{
	size_t blockSize = get_local_size(0);
	size_t tid = get_local_id(0);
	size_t i = get_group_id(0) * blockSize + get_local_id(0);
	size_t gridSize = blockSize*get_num_groups(0);
	float result;
	
	if (i < n)
	{
		result = input[i];
		i += gridSize;
	}
	
	while (i < n)
	{
		result = scalarAddFunc(result, input[i]);
		i += gridSize;
	}
	
	sdata[tid] = result;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (blockSize >= 1024) { if (tid < 512 && tid + 512 < n) { sdata[tid] = scalarAddFunc(sdata[tid], sdata[tid + 512]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=  512) { if (tid < 256 && tid + 256 < n) { sdata[tid] = scalarAddFunc(sdata[tid], sdata[tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=  256) { if (tid < 128 && tid + 128 < n) { sdata[tid] = scalarAddFunc(sdata[tid], sdata[tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=  128) { if (tid <  64 && tid +  64 < n) { sdata[tid] = scalarAddFunc(sdata[tid], sdata[tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=   64) { if (tid <  32 && tid +  32 < n) { sdata[tid] = scalarAddFunc(sdata[tid], sdata[tid +  32]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=   32) { if (tid <  16 && tid +  16 < n) { sdata[tid] = scalarAddFunc(sdata[tid], sdata[tid +  16]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=   16) { if (tid <   8 && tid +   8 < n) { sdata[tid] = scalarAddFunc(sdata[tid], sdata[tid +   8]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=    8) { if (tid <   4 && tid +   4 < n) { sdata[tid] = scalarAddFunc(sdata[tid], sdata[tid +   4]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=    4) { if (tid <   2 && tid +   2 < n) { sdata[tid] = scalarAddFunc(sdata[tid], sdata[tid +   2]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (blockSize >=    2) { if (tid <   1 && tid +   1 < n) { sdata[tid] = scalarAddFunc(sdata[tid], sdata[tid +   1]); } barrier(CLK_LOCAL_MEM_FENCE); }
	
	if (tid == 0)
		output[get_group_id(0)] = sdata[tid];
}
)###");
		
		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu2::backend::Device_CL *device : skepu2::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu2::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel = clCreateKernel(program, "dotproduct_precompiled_ReduceKernel_scalarAddFunc", &err);
			CL_CHECK_ERROR(err, "Error creating map kernel '" << "dotproduct_precompiled_ReduceKernel_scalarAddFunc" << "'");
			
			kernels(counter++, &kernel);
		}
		
		initialized = true;
	}
	
	static void reduce(size_t deviceID, size_t localSize, size_t globalSize, cl_mem input, cl_mem output, size_t n, size_t sharedMemSize)
	{
		skepu2::backend::cl_helpers::setKernelArgs(kernels(deviceID), input, output, n);
		clSetKernelArg(kernels(deviceID), 3, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernels(deviceID), 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Map kernel");
	}
};
