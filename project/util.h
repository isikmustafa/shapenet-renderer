#pragma once

#include <iostream>
#include <functional>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR( err ) (util::checkCudaError( err, __FILE__, __LINE__))

namespace util
{
	__host__ inline void checkCudaError(cudaError_t err, const char* file, int line)
	{
		if (err != cudaSuccess)
		{
			std::cerr << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
			system("PAUSE");
		}
	}
}