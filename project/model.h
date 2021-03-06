#pragma once

#include "device_array.h"
#include "triangle.h"
#include "bvh.h"
#include "material.h"

#include <string>

class Model
{
public:
	Model(const std::string& path);

	const Bbox& getBbox() const { return m_bbox; }

#ifdef __NVCC__
	__device__ bool intersect(const Ray& ray, Intersection& intersection, float max_distance)
	{
		auto triangles = m_triangles.getPtr();
		auto nodes = m_nodes.getPtr();

		auto min_distance = max_distance;

		auto ray_origin = ray.getOrigin();
		auto ray_inv_dir = 1.0f / ray.getDirection();
		auto result = m_bbox.intersect(ray_origin, ray_inv_dir);
		if (result <= 0.0f || result >= min_distance)
		{
			return false;
		}

		int triangle_index = -1;
		BVHNode* traversal_stack[32];
		int traversal_stack_top = -1;
		BVHNode* intersection_stack[16];
		int intersection_stack_top = -1;

		//inner node
		if (nodes[0].left_node)
		{
			traversal_stack[++traversal_stack_top] = &nodes[0];
		}
		//leaf node
		else
		{
			intersection_stack[++intersection_stack_top] = &nodes[0];
		}

		while (traversal_stack_top >= 0)
		{
			auto node = traversal_stack[traversal_stack_top--];

			for (int child_index = 0; child_index <= 1; ++child_index)
			{
				result = reinterpret_cast<Bbox*>(node)[child_index].intersect(ray_origin, ray_inv_dir);
				if (result > 0.0f && result < min_distance)
				{
					//inner node
					constexpr int stride = 2 * sizeof(Bbox) / sizeof(unsigned int);
					auto child_node = reinterpret_cast<unsigned int*>(node)[stride + child_index];
					if (nodes[child_node].left_node)
					{
						traversal_stack[++traversal_stack_top] = &nodes[child_node];
					}
					//leaf node
					else
					{
						intersection_stack[++intersection_stack_top] = &nodes[child_node];
					}
				}
			}

			while (__all_sync(0xFFFFFFFF, intersection_stack_top >= 0))
			{
				auto intersected_ptr = intersection_stack[intersection_stack_top--];
				for (int i = intersected_ptr->start_index; i <= intersected_ptr->end_index; ++i)
				{
					auto result = triangles[i].intersectShadowRay(ray);
					if (result > 0.0f && result < min_distance)
					{
						min_distance = result;
						triangle_index = i;
					}
				}
			}

			if (intersection_stack_top >= 14)
			{
				auto intersected_ptr = intersection_stack[intersection_stack_top--];
				for (int i = intersected_ptr->start_index; i <= intersected_ptr->end_index; ++i)
				{
					auto result = triangles[i].intersectShadowRay(ray);
					if (result > 0.0f && result < min_distance)
					{
						min_distance = result;
						triangle_index = i;
					}
				}
			}
		}

		while (intersection_stack_top >= 0)
		{
			auto intersected_ptr = intersection_stack[intersection_stack_top--];
			for (int i = intersected_ptr->start_index; i <= intersected_ptr->end_index; ++i)
			{
				auto result = triangles[i].intersectShadowRay(ray);
				if (result > 0.0f && result < min_distance)
				{
					min_distance = result;
					triangle_index = i;
				}
			}
		}

		if (triangle_index >= 0)
		{
			triangles[triangle_index].intersect(ray, intersection);

			return true;
		}

		return false;
	}

	__device__ const util::DeviceArray<Material>& getMaterials() const { return m_materials; }
#endif

private:
	util::DeviceArray<Triangle> m_triangles;
	util::DeviceArray<BVHNode> m_nodes;
	util::DeviceArray<Material> m_materials;
	Bbox m_bbox;
};