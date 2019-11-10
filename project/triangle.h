#pragma once

#include "ray.h"
#include "intersection.h"
#include "bbox.h"

#include <glm/glm.hpp>
#include <cuda_runtime.h>

class Triangle
{
public:
	Triangle(const glm::vec3& v0, const glm::vec3& edge1, const glm::vec3& edge2,
		const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2,
		const glm::vec2& t0, const glm::vec2& t1, const glm::vec2& t2,
		int material_id);

	Bbox getBbox() const;

#ifdef __NVCC__
	__device__ bool intersect(const Ray& ray, Intersection& intersection) const
	{
		//Möller-Trumbore algorithm
		auto pvec = glm::cross(ray.getDirection(), m_edge2);
		auto inv_det = 1.0f / glm::dot(m_edge1, pvec);

		auto tvec = ray.getOrigin() - m_v0;
		auto w1 = glm::dot(tvec, pvec) * inv_det;

		if (w1 < 0.0f || w1 > 1.0f)
		{
			return false;
		}

		auto qvec = glm::cross(tvec, m_edge1);
		auto w2 = glm::dot(ray.getDirection(), qvec) * inv_det;

		if (w2 < 0.0f || (w1 + w2) > 1.0f)
		{
			return false;
		}

		//Fill the intersection record.
		auto w0 = (1.0f - w1 - w2);
		intersection.normal = glm::normalize(w0 * m_n0 + w1 * m_n1 + w2 * m_n2);
		intersection.tex_coord = w0 * m_t0 + w1 * m_t1 + w2 * m_t2;
		intersection.distance = glm::dot(m_edge2, qvec) * inv_det;
		intersection.material_id = m_material_id;

		return true;
	}

	__device__ float intersectShadowRay(const Ray& ray) const
	{
		//Möller-Trumbore algorithm
		auto pvec = glm::cross(ray.getDirection(), m_edge2);
		auto inv_det = 1.0f / glm::dot(m_edge1, pvec);

		auto tvec = ray.getOrigin() - m_v0;
		auto w1 = glm::dot(tvec, pvec) * inv_det;

		if (w1 < 0.0f || w1 > 1.0f)
		{
			return -1.0f;
		}

		auto qvec = glm::cross(tvec, m_edge1);
		auto w2 = glm::dot(ray.getDirection(), qvec) * inv_det;

		if (w2 < 0.0f || (w1 + w2) > 1.0f)
		{
			return -1.0f;
		}

		return glm::dot(m_edge2, qvec) * inv_det;
	}
#endif

private:
	//Positions
	glm::vec3 m_v0;
	glm::vec3 m_edge1;
	glm::vec3 m_edge2;

	//Normals
	glm::vec3 m_n0;
	glm::vec3 m_n1;
	glm::vec3 m_n2;

	//Texcoords
	glm::vec2 m_t0;
	glm::vec2 m_t1;
	glm::vec2 m_t2;

	//Material
	int m_material_id;
};