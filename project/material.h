#pragma once

#include "texture.h"

#include <glm/glm.hpp>

class Material
{
public:
	Material(const glm::vec3& ambient, const glm::vec3& diffuse,
		const std::string& ambient_texture_filename, const std::string& diffuse_texture_filename);

#ifdef __NVCC__
	__device__ glm::vec3 fetchAmbient(const glm::vec2& tex_coord) const
	{
		return m_ambient * m_ambient_texture.fetch(tex_coord);
	}

	__device__ glm::vec3 fetchDiffuse(const glm::vec2& tex_coord) const
	{
		return m_diffuse * m_diffuse_texture.fetch(tex_coord);
	}
#endif

private:
	glm::vec3 m_ambient;
	glm::vec3 m_diffuse;

	Texture m_ambient_texture;
	Texture m_diffuse_texture;
};