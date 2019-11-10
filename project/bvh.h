//From https://github.com/isikmustafa/pathtracer/blob/master/bvh.cuh

#pragma once

#include "bbox.h"

#include <vector>

class Mesh;
class Triangle;

__align__(64)
struct BVHNode
{
	Bbox left_bbox;
	Bbox right_bbox;
	unsigned int left_node; //zero for leaf nodes
	unsigned int right_node; //zero for leaf nodes
	union
	{
		//Mesh
		struct
		{
			unsigned int start_index; //zero for inner nodes
			unsigned int end_index; //zero for inner nodes
		};
		//Scene
		struct
		{
			unsigned int mesh_id; //zero for inner nodes
			unsigned int __pad__;
		};
	};

	BVHNode(const Bbox& p_left_bbox, const Bbox& p_right_bbox, unsigned int p_left_node, unsigned int p_right_node, unsigned int p_start_index, unsigned int p_end_index)
		: left_bbox(p_left_bbox)
		, right_bbox(p_right_bbox)
		, left_node(p_left_node)
		, right_node(p_right_node)
		, start_index(p_start_index)
		, end_index(p_end_index)
	{}
};

void buildMeshBvh(std::vector<Triangle>& mesh_triangles, std::vector<BVHNode>& nodes);