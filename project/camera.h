#pragma once

#include <glm/glm.hpp>
#include <string>
#include <cuda_runtime.h>

struct Camera
{
	Camera(const glm::vec3& p_look_at, const glm::vec3& p_position);

#ifdef __NVCC__
	__device__ glm::vec3 pointEyeToWorld(const glm::vec3& point) const
	{
		return rotation * point + position;
	}

	__device__ glm::vec3 vectorEyeToWorld(const glm::vec3& vector) const
	{
		return rotation * vector;
	}
#endif

	//TODO: Implement
	void dumpToFile(const std::string& filename) const;

	glm::vec3 position;
	glm::mat3 rotation;
};