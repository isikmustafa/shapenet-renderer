#include "camera.h"

Camera::Camera(const glm::vec3& p_look_at, const glm::vec3& p_position)
	: position(p_position)
{
	glm::vec3 forward = glm::normalize(p_look_at - position);
	glm::vec3 up(0.0f, -1.0f, 0.0f);
	glm::vec3 right = glm::cross(up, forward);
	up = glm::cross(forward, right);

	rotation = glm::mat3(right, up, forward);
}
