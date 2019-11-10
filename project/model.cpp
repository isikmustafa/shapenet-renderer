#include "model.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

Model::Model(const std::string& path)
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;
	std::string warn;

	auto ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, (path + "model_normalized.obj").c_str(), path.c_str(), true, true);

	if (!warn.empty())
	{
		std::cout << warn << std::endl;
	}

	if (!err.empty())
	{
		std::cerr << err << std::endl;
	}

	if (!ret)
	{
		throw std::runtime_error("Error: Model cannot be loaded.");
	}

	//Create materials
	for (const auto& mat : materials)
	{

	}

	//Loop over shapes.
	std::vector<Triangle> triangles;
	int shapes_size = shapes.size();
	for (int s = 0; s < shapes_size; ++s)
	{
		int index_offset = 0;

		// Loop over vertices in the face.
		int num_of_faces = shapes[s].mesh.num_face_vertices.size();
		for (int f = 0; f < num_of_faces; ++f)
		{
			if (shapes[s].mesh.num_face_vertices[f] != 3)
			{
				throw std::exception("Error: There is a polygon which is not triangle!");
			}

			//Loop over vertices in the face.
			glm::vec3 positions[3];
			glm::vec3 normals[3];
			glm::vec2 tex_coords[3];
			for (int v = 0; v < 3; ++v)
			{
				// access to vertex
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
				positions[v].x = attrib.vertices[3 * idx.vertex_index + 0];
				positions[v].y = attrib.vertices[3 * idx.vertex_index + 1];
				positions[v].z = attrib.vertices[3 * idx.vertex_index + 2];
				normals[v].x = attrib.normals[3 * idx.normal_index + 0];
				normals[v].y = attrib.normals[3 * idx.normal_index + 1];
				normals[v].z = attrib.normals[3 * idx.normal_index + 2];
				tex_coords[v].x = attrib.texcoords[2 * idx.texcoord_index + 0];
				tex_coords[v].y = attrib.texcoords[2 * idx.texcoord_index + 1];
			}
			index_offset += 3;

			int material_id = shapes[s].mesh.material_ids[f];

			m_bbox.extend(positions[0]);
			m_bbox.extend(positions[1]);
			m_bbox.extend(positions[2]);

			triangles.push_back(Triangle(positions[0], positions[1] - positions[0], positions[2] - positions[0],
				normals[0], normals[1], normals[2], tex_coords[0], tex_coords[1], tex_coords[2], material_id));
		}
	}

	std::vector<BVHNode> bvh_nodes;
	buildMeshBvh(triangles, bvh_nodes);

	m_triangles = util::DeviceArray<Triangle>(triangles);
	m_nodes = util::DeviceArray<BVHNode>(bvh_nodes);
}
