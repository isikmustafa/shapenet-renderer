#pragma once

#include "model.h"
#include "camera.h"

void raytracer(Model* model, const Camera& camera, cudaSurfaceObject_t output, int width, int height);