#ifndef SKEPU_PRECOMPILED
#define SKEPU_PRECOMPILED
#endif
#ifndef SKEPU_OPENMP
#define SKEPU_OPENMP
#endif
#ifndef SKEPU_OPENCL
#define SKEPU_OPENCL
#endif
/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>
#include <iterator>

#include <skepu2.hpp>

#include "support.h"


unsigned char median_kernel(int ox, int oy, size_t stride, const unsigned char *image, size_t elemPerPx)
{
	// your code here
	int frequency[256] = {0};	// This array will store the frequency of occured pixel values
	int numPixels = ox * oy;
	int pixelCounter = 0;
	int median;
	
	// Loop over pixels and store pixel value frequencies in the array
	for (int y = -oy; y <= oy; ++y)
		for (int x = -ox; x <= ox; x += elemPerPx)
			frequency[image[y*(int)stride+x]]++;

	int prevValue = 0;
	for(int i = 0; i < 256; i++) {
		// Check if we have traversed half of the pixels (i.e. where the median should be)
		// Since a lot of values will be zero in frequency, we need to count
		// the number of elements actually containing "real" information
		pixelCounter += frequency[i];
		if(pixelCounter > numPixels/2) {
			median = (prevValue != 0) ? (i + prevValue)/2 : i;
			break;
		}
		prevValue = (frequency[i] != 0) ? i : prevValue;
	}
	return median;
}
struct skepu2_userfunction_calculateMedian_median_kernel
{
constexpr static size_t totalArity = 5;
constexpr static bool indexed = 0;
using ElwiseArgs = std::tuple<int, int, size_t, const unsigned char *>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<size_t>;
constexpr static skepu2::AccessMode anyAccessMode[] = {
};

using Ret = unsigned char;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char OMP(int ox, int oy, size_t stride, const unsigned char *image, size_t elemPerPx)
{
	// your code here
	int frequency[256] = {0};	// This array will store the frequency of occured pixel values
	int numPixels = ox * oy;
	int pixelCounter = 0;
	int median;
	
	// Loop over pixels and store pixel value frequencies in the array
	for (int y = -oy; y <= oy; ++y)
		for (int x = -ox; x <= ox; x += elemPerPx)
			frequency[image[y*(int)stride+x]]++;

	int prevValue = 0;
	for(int i = 0; i < 256; i++) {
		// Check if we have traversed half of the pixels (i.e. where the median should be)
		// Since a lot of values will be zero in frequency, we need to count
		// the number of elements actually containing "real" information
		pixelCounter += frequency[i];
		if(pixelCounter > numPixels/2) {
			median = (prevValue != 0) ? (i + prevValue)/2 : i;
			break;
		}
		prevValue = (frequency[i] != 0) ? i : prevValue;
	}
	return median;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char CPU(int ox, int oy, size_t stride, const unsigned char *image, size_t elemPerPx)
{
	// your code here
	int frequency[256] = {0};	// This array will store the frequency of occured pixel values
	int numPixels = ox * oy;
	int pixelCounter = 0;
	int median;
	
	// Loop over pixels and store pixel value frequencies in the array
	for (int y = -oy; y <= oy; ++y)
		for (int x = -ox; x <= ox; x += elemPerPx)
			frequency[image[y*(int)stride+x]]++;

	int prevValue = 0;
	for(int i = 0; i < 256; i++) {
		// Check if we have traversed half of the pixels (i.e. where the median should be)
		// Since a lot of values will be zero in frequency, we need to count
		// the number of elements actually containing "real" information
		pixelCounter += frequency[i];
		if(pixelCounter > numPixels/2) {
			median = (prevValue != 0) ? (i + prevValue)/2 : i;
			break;
		}
		prevValue = (frequency[i] != 0) ? i : prevValue;
	}
	return median;
}
#undef SKEPU_USING_BACKEND_CPU
};

constexpr skepu2::AccessMode skepu2_userfunction_calculateMedian_median_kernel::anyAccessMode[];





#include "median_precompiled_Overlap2DKernel_median_kernel_cl_source.inl"
int main(int argc, char* argv[])
{
	LodePNGColorType colorType = LCT_RGB;

	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << "input output radius [backend]\n";
		exit(1);
	}

	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[4])};

	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFileNamePad = outputFileName + ss.str() + "-median.png";

	// Read the padded image into a matrix. Create the output matrix without padding.
	ImageInfo imageInfo;
	skepu2::Matrix<unsigned char> inputMatrix = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu2::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);

	// Skeleton instance
	skepu2::backend::MapOverlap2D<skepu2_userfunction_calculateMedian_median_kernel, bool, CLWrapperClass_median_precompiled_Overlap2DKernel_median_kernel> calculateMedian(false);
	calculateMedian.setBackend(spec);
	calculateMedian.setOverlap(radius, radius  * imageInfo.elementsPerPixel);

	auto timeTaken = skepu2::benchmark::measureExecTime([&]
	{
		calculateMedian(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);
	});

	WritePngFileMatrix(outputMatrix, outputFileNamePad, colorType, imageInfo);

	std::cout << "Time: " << (timeTaken.count() / 10E6) << "\n";

	return 0;
}
