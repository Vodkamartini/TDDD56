/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <iostream>

#include <skepu2.hpp>

/* SkePU user functions */

float mult(float a, float b)
{
	return a*b;
}

float add(float a, float b)
{
	return a+b;
}


int main(int argc, const char* argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <input size> <backend>\n";
		exit(1);
	}

	const size_t size = std::stoul(argv[1]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[2])};
//	spec.setCPUThreads(<integer value>);


	/* Skeleton instances */
//	auto instance = skepu2::Map<???>(userfunction);
// ...

	/* Set backend (important, do for all instances!) */
//	instance.setBackend(spec);

	// Separate Map + Reduce
	auto dotProductMap = skepu2::Map<2>(mult);
	dotProductMap.setBackend(spec);
	auto dotProductReduce = skepu2::Reduce(add);
	dotProductReduce.setBackend(spec);

	// Combined MapReduce
	auto dotProduct = skepu2::MapReduce<2>(mult, add);
	dotProduct.setBackend(spec);

	/* SkePU containers */
	skepu2::Vector<float> v1(size, 1.0f), v2(size, 2.0f);
	skepu2::Vector<float> v3(size, 0.0f);

	/* Compute and measure time */
	float resComb, resSep;

	auto timeComb = skepu2::benchmark::measureExecTime([&]
	{
		// your code here
		resComb = dotProduct(v1, v2);
	});

	auto timeSep = skepu2::benchmark::measureExecTime([&]
	{
		// your code here
		dotProductMap(v3, v1, v2);
		resSep = dotProductReduce(v3);
	});

	std::cout << "Time Combined: " << (timeComb.count() / 10E6) << " seconds.\n";
	std::cout << "Time Separate: " << ( timeSep.count() / 10E6) << " seconds.\n";


	std::cout << "Result Combined: " << resComb << "\n";
	std::cout << "Result Separate: " << resSep  << "\n";

	return 0;
}
