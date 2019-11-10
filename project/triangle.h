#pragma once

#include <glm/glm.hpp>

class Triangle
{
public:


private:
	//Positions
	glm::vec3 m_v0;
	glm::vec3 m_v1;
	glm::vec3 m_v2;

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