#include <iostream>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <glm/glm.hpp>
#include "texture.h"

int main()
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;
	std::string warn;

	const std::string path = "C:/Users/Mustafa/Downloads/ShapeNetCore.v2/02958343/1a1dcd236a1e6133860800e6696b8284/models/";
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

	Texture tex("C:/Users/Mustafa/Downloads/ShapeNetCore.v2/02958343/1a1dcd236a1e6133860800e6696b8284/images/texture1.jpg");

	//Create materials
	for (const auto& mat : materials)
	{

	}

	//Loop over shapes.
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
			for (int v = 0; v < 3; ++v)
			{
				// access to vertex
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
				tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
				tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
				tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
				tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
				tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
				tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];
				tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index + 0];
				tinyobj::real_t ty = attrib.texcoords[2 * idx.texcoord_index + 1];
			}
			index_offset += 3;

			// per-face material
			shapes[s].mesh.material_ids[f];
		}
	}

	return 0;
}
