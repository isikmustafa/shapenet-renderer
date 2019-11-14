#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

struct Bbox
{
public:
	Bbox();
	Bbox(const glm::vec3& p_min, const glm::vec3& p_max);

	void extend(const glm::vec3& point);
	void extend(const Bbox& bbox);
	float getSurfaceArea() const;
	glm::vec3 getCenter() const;

	const glm::vec3& getMin() const { return m_min; }
	const glm::vec3& getMax() const { return m_max; }

#ifdef __NVCC__
	__device__ float intersect(const glm::vec3& origin, const glm::vec3& inv_ray_dir) const
	{
		auto t0 = (m_min - origin) * inv_ray_dir;
		auto t1 = (m_max - origin) * inv_ray_dir;

		auto min = glm::min(t0, t1);
		auto max = glm::max(t0, t1);
		auto tmin = glm::max(min.x, glm::max(min.y, min.z));
		auto tmax = glm::min(max.x, glm::min(max.y, max.z));

		if (tmax < tmin)
		{
			return -1.0f;
		}

		return tmin > 0.0f ? tmin : tmax;
	}
#endif

private:
	glm::vec3 m_min;
	glm::vec3 m_max;
};