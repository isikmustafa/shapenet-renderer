#include "bbox.h"

#include <numeric>

Bbox::Bbox()
	: m_min(std::numeric_limits<float>::max())
	, m_max(-std::numeric_limits<float>::max())
{}

Bbox::Bbox(const glm::vec3& p_min, const glm::vec3& p_max)
	: m_min(p_min)
	, m_max(p_max)
{}

void Bbox::extend(const glm::vec3& point)
{
	m_min = glm::min(m_min, point);
	m_max = glm::max(m_max, point);
}

void Bbox::extend(const Bbox& bbox)
{
	m_min = glm::min(m_min, bbox.getMin());
	m_max = glm::max(m_max, bbox.getMax());
}

float Bbox::getSurfaceArea() const
{
	auto edges = m_max - m_min;
	return 2 * (edges.x * edges.y + edges.x * edges.z + edges.y * edges.z);
}