#include "material.h"

Material::Material(const glm::vec3& ambient, const glm::vec3& diffuse,
	const std::string& ambient_texture_filename, const std::string& diffuse_texture_filename)
	: m_ambient(ambient)
	, m_diffuse(diffuse)
	, m_ambient_texture(ambient_texture_filename)
	, m_diffuse_texture(diffuse_texture_filename)
{}
