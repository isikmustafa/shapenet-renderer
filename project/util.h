#pragma once

#include <iostream>
#include <functional>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/trigonometric.hpp>

#define CHECK_CUDA_ERROR( err ) (util::checkCudaError( err, __FILE__, __LINE__))

namespace util
{
	inline void checkCudaError(cudaError_t err, const char* file, int line)
	{
		if (err != cudaSuccess)
		{
			std::cerr << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
			system("PAUSE");
		}
	}

	inline float runKernelGetExecutionTime(std::function<void(void)> kernel)
	{
		cudaEvent_t start, end;
		CHECK_CUDA_ERROR(cudaEventCreate(&start));
		CHECK_CUDA_ERROR(cudaEventCreate(&end));

		CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
		kernel();
		CHECK_CUDA_ERROR(cudaEventRecord(end, 0));
		CHECK_CUDA_ERROR(cudaEventSynchronize(end));

		float elapsed_time = 0.0f;
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time, start, end));

		CHECK_CUDA_ERROR(cudaEventDestroy(start));
		CHECK_CUDA_ERROR(cudaEventDestroy(end));

		return elapsed_time;
	}

	inline glm::vec3 sampleHemisphereUniform(float u1, float u2, float radius)
	{
		auto theta = glm::acos(u1);
		auto phi = glm::two_pi<float>() * u2;

		auto sin_theta = glm::sin(theta);
		return glm::vec3(glm::cos(phi) * sin_theta, glm::sin(phi) * sin_theta, glm::cos(theta)) * radius;
	}

	inline glm::vec3 sampleSphereUniform(float u1, float u2, float radius)
	{
		auto theta = glm::acos(1.0f - 2.0f * u1);
		auto phi = glm::two_pi<float>() * u2;

		auto sin_theta = glm::sin(theta);
		return glm::vec3(glm::cos(phi) * sin_theta, glm::sin(phi) * sin_theta, glm::cos(theta)) * radius;
	}
}