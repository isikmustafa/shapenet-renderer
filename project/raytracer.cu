#pragma once

#include "raytracer.h"
#include "device_util.h"

__global__ void raytracerKernel(Model* model, Camera camera, glm::vec3 light_direction, cudaSurfaceObject_t output, int width, int height)
{
	auto index = util::getThreadIndex2D();
	if (index.x >= width || index.y >= height)
	{
		return;
	}

	glm::vec3 uv(index.x + 0.5f, index.y + 0.5f, 1.0f);

	auto eye_coord = camera.inverse_intrinsics * uv;
	auto world_coord = camera.pointEyeToWorld(eye_coord);

	Ray ray(camera.position, glm::normalize(glm::vec3(world_coord) - camera.position));

	glm::vec3 color(255.0f);
	Intersection intersection;

	if (model->intersect(ray, intersection, FLT_MAX))
	{
		auto materials = model->getMaterials().getPtr();
		color = materials[intersection.material_id].fetchAmbient(intersection.tex_coord);
		color += materials[intersection.material_id].fetchDiffuse(intersection.tex_coord) * glm::max(0.0f, glm::dot(-light_direction, intersection.normal));
		color *= 255.0f;

		//color = glm::abs(intersection.normal) * 255.0f;
	}

	//BGR is intentionally given.
	color = glm::clamp(color, 0.0f, 255.0f);
	surf2Dwrite(util::rgbToUint({ color.r, color.g, color.b }), output, index.x * 4, index.y);
}

void raytracer(Model* model, const Camera& camera, const glm::vec3& light_direction, cudaSurfaceObject_t output, int width, int height)
{
	dim3 threads(8, 8);
	dim3 blocks(width / threads.x + 1, height / threads.y + 1);
	raytracerKernel << <blocks, threads >> > (model, camera, light_direction, output, width, height);
}