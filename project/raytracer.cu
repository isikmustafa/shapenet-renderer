#pragma once

#include "raytracer.h"
#include "device_util.h"

__global__ void raytracerKernel(Model* model, cudaSurfaceObject_t output, int width, int height)
{
	auto index = util::getThreadIndex2D();
	if (index.x >= width || index.y >= height)
	{
		return;
	}

	glm::vec3 cam_position(0.0, 0.0f, -1.3f);
	glm::mat3 intrinsics(glm::vec3(525.0f, 0.0f, 0.0f), glm::vec3(0.0f, 525.0f, 0.0f), glm::vec3(256.0f, 256.0f, 1.0f));
	glm::vec3 uv(index.x + 0.5f, height - index.y + 0.5f, 1.0f);
	auto eye_coord = glm::inverse(intrinsics) * uv;
	auto world_coord = eye_coord + cam_position;
	Ray ray(cam_position, glm::normalize(world_coord - cam_position));

	glm::vec3 color(0.0f);
	Intersection intersection;

	if (model->intersect(ray, intersection, FLT_MAX))
	{
		color = intersection.normal * 255.0f;
	}

	surf2Dwrite(util::rgbToUint({ color.x, color.y, color.z }), output, index.x * 4, index.y);
}

void raytracer(Model* model, cudaSurfaceObject_t output, int width, int height)
{
	dim3 threads(16, 16);
	dim3 blocks(width / threads.x + 1, height / threads.y + 1);
	raytracerKernel << <blocks, threads >> > (model, output, width, height);
}