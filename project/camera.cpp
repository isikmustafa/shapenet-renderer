#include "camera.h"

#include <fstream>

Camera::Camera(const glm::vec3& p_look_at, const glm::vec3& p_position)
	: position(p_position)
{
	glm::vec3 forward = glm::normalize(p_look_at - position);
	glm::vec3 up(0.0f, -1.0f, 0.0f);
	glm::vec3 right = glm::cross(up, forward);
	up = glm::cross(forward, right);

	rotation = glm::mat3(right, up, forward);
}

void Camera::dumpToFile(const std::string& filename) const
{
	std::ofstream out(filename);

	out << rotation[0][0] << " " << rotation[1][0] << " " << rotation[2][0] << " " << position.x << " ";
	out << rotation[0][1] << " " << rotation[1][1] << " " << rotation[2][1] << " " << position.y << " ";
	out << rotation[0][2] << " " << rotation[1][2] << " " << rotation[2][2] << " " << position.z << " ";
	out << 0.0f << " " << 0.0f << " " << 0.0f << " " << 1.0f << std::endl;

	out.close();
}
