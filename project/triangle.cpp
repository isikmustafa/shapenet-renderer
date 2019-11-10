#include "triangle.h"

Triangle::Triangle(const glm::vec3& v0, const glm::vec3& edge1, const glm::vec3& edge2,
	const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2,
	const glm::vec2& t0, const glm::vec2& t1, const glm::vec2& t2,
	int material_id)
	: m_v0(v0)
	, m_edge1(edge1)
	, m_edge2(edge2)
	, m_n0(n0)
	, m_n1(n1)
	, m_n2(n2)
	, m_t0(t0)
	, m_t1(t1)
	, m_t2(t2)
	, m_material_id(material_id)
{}

Bbox Triangle::getBbox() const
{
	auto v1 = m_edge1 + m_v0;
	auto v2 = m_edge2 + m_v0;

	return Bbox(glm::min(glm::min(v1, v2), m_v0), glm::max(glm::max(v1, v2), m_v0));
}
