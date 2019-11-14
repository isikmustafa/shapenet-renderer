#pragma once

#include "model.h"
#include "camera.h"

void raytracer(Model* model, const Camera& camera, const glm::vec3& light_direction, cudaSurfaceObject_t output, int width, int height);