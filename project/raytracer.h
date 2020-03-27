#pragma once

#include "model.h"
#include "camera.h"

struct SHCoeffs
{
	glm::vec3 l00{ 0.0f, 0.0f, 0.0f };

	glm::vec3 l1_1{ 0.0f, 0.0f, 0.0f };
	glm::vec3 l10{ 0.0f, 0.0f, 0.0f };
	glm::vec3 l11{ 0.0f, 0.0f, 0.0f };

	glm::vec3 l2_2{ 0.0f, 0.0f, 0.0f };
	glm::vec3 l2_1{ 0.0f, 0.0f, 0.0f };
	glm::vec3 l20{ 0.0f, 0.0f, 0.0f };
	glm::vec3 l21{ 0.0f, 0.0f, 0.0f };
	glm::vec3 l22{ 0.0f, 0.0f, 0.0f };
};

bool raytracer(Model* model, const Camera& camera, const SHCoeffs& sh_coeffs, const glm::vec3& light_direction, cudaSurfaceObject_t output, int width, int height);