#pragma once

#include <glm/glm.hpp>

struct Intersection
{
	glm::vec3 normal;
	glm::vec2 tex_coord;
	float distance;
	unsigned int material_id;
};