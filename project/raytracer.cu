#pragma once

#include "raytracer.h"
#include "device_util.h"

__global__ void raytracerKernel(Model* model, Camera camera, cudaSurfaceObject_t output, int width, int height)
{
	auto index = util::getThreadIndex2D();
	if (index.x >= width || index.y >= height)
	{
		return;
	}

	glm::mat3 intrinsics(glm::vec3(525.0f, 0.0f, 0.0f), glm::vec3(0.0f, 525.0f, 0.0f), glm::vec3(256.0f, 256.0f, 1.0f));
	glm::vec3 uv(index.x + 0.5f, index.y + 0.5f, 1.0f);

	auto eye_coord = glm::inverse(intrinsics) * uv;
	auto world_coord = camera.pointEyeToWorld(eye_coord);

	Ray ray(camera.position, glm::normalize(glm::vec3(world_coord) - camera.position));

	glm::vec3 color(255.0f);
	Intersection intersection;

	if (model->intersect(ray, intersection, FLT_MAX))
	{
		auto materials = model->getMaterials().getPtr();
		color = materials[intersection.material_id].fetchAmbient(intersection.tex_coord);
		color += materials[intersection.material_id].fetchDiffuse(intersection.tex_coord) * glm::dot(-glm::normalize(glm::vec3(0.0f, -1.0f, 1.0f)), intersection.normal);
		color *= 255.0f;

		//color = glm::abs(intersection.normal) * 255.0f;
	}

	//BGR is intentionally given.
	surf2Dwrite(util::rgbToUint({ color.b, color.g, color.r }), output, index.x * 4, index.y);
}

void raytracer(Model* model, const Camera& camera, cudaSurfaceObject_t output, int width, int height)
{
	dim3 threads(16, 16);
	dim3 blocks(width / threads.x + 1, height / threads.y + 1);
	raytracerKernel << <blocks, threads >> > (model, camera, output, width, height);
}