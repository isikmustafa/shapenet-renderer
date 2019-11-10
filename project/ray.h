#pragma once

#include <glm/glm.hpp>

#include <cuda_runtime.h>

class Ray
{
public:
#ifdef __NVCC__
	__device__ Ray(const glm::vec3& origin, const glm::vec3& direction)
		: m_origin(origin)
		, m_direction(direction)
	{}

	__device__ glm::vec3 getPoint(float distance) const
	{
		return m_origin + distance * m_direction;
	}

	__device__ const glm::vec3& getOrigin() const { return m_origin; }

	__device__ const glm::vec3& getDirection() const { return m_direction; }
#endif

private:
	glm::vec3 m_origin;
	glm::vec3 m_direction;
};