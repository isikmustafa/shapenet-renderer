#pragma once

#include "device_array.h"
#include "triangle.h"
#include "bvh.h"

#include <string>

class Model
{
public:
	Model(const std::string& path);

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
		BVHNode* intersected_ptr = nullptr;

		//inner node
		if (nodes[0].left_node)
		{
			traversal_stack[++traversal_stack_top] = &nodes[0];
		}
		//leaf node
		else
		{
			intersected_ptr = &nodes[0];
		}

		while (traversal_stack_top >= 0)
		{
			auto node = traversal_stack[traversal_stack_top--];

			result = node->left_bbox.intersect(ray_origin, ray_inv_dir);
			if (result > 0.0f && result < min_distance)
			{
				//inner node
				if (nodes[node->left_node].left_node)
				{
					traversal_stack[++traversal_stack_top] = &nodes[node->left_node];
				}
				//leaf node
				else
				{
					if (intersected_ptr)
					{
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
					intersected_ptr = &nodes[node->left_node];
				}
			}
			result = node->right_bbox.intersect(ray_origin, ray_inv_dir);
			if (result > 0.0f && result < min_distance)
			{
				//inner node
				if (nodes[node->right_node].right_node)
				{
					traversal_stack[++traversal_stack_top] = &nodes[node->right_node];
				}
				//leaf node
				else
				{
					if (intersected_ptr)
					{
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
					intersected_ptr = &nodes[node->right_node];
				}
			}

			if (__all(intersected_ptr != nullptr))
			{
				for (int i = intersected_ptr->start_index; i <= intersected_ptr->end_index; ++i)
				{
					auto result = triangles[i].intersectShadowRay(ray);
					if (result > 0.0f && result < min_distance)
					{
						min_distance = result;
						triangle_index = i;
					}
				}

				intersected_ptr = nullptr;
			}
		}

		if (intersected_ptr)
		{
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
#endif

private:
	util::DeviceArray<Triangle> m_triangles;
	util::DeviceArray<BVHNode> m_nodes;
	Bbox m_bbox;
};