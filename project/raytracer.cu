#pragma once

#include "raytracer.h"
#include "device_util.h"

__device__ glm::mat4 buildM(const SHCoeffs& coeffs, int axis)
{
	auto c1 = 0.429043f;
	auto c2 = 0.511664f;
	auto c3 = 0.743125f;
	auto c4 = 0.886227f;
	auto c5 = 0.247708f;

	glm::mat4 m;

	//0th column
	m[0][0] = c1 * coeffs.l22[axis];
	m[0][1] = c1 * coeffs.l2_2[axis];
	m[0][2] = c1 * coeffs.l21[axis];
	m[0][3] = c2 * coeffs.l11[axis];

	//1st column
	m[1][0] = c1 * coeffs.l2_2[axis];
	m[1][1] = -c1 * coeffs.l22[axis];
	m[1][2] = c1 * coeffs.l2_1[axis];
	m[1][3] = c2 * coeffs.l1_1[axis];

	//2nd column
	m[2][0] = c1 * coeffs.l21[axis];
	m[2][1] = c1 * coeffs.l2_1[axis];
	m[2][2] = c3 * coeffs.l20[axis];
	m[2][3] = c2 * coeffs.l10[axis];

	//3rd column
	m[3][0] = c2 * coeffs.l11[axis];
	m[3][1] = c2 * coeffs.l1_1[axis];
	m[3][2] = c2 * coeffs.l10[axis];
	m[3][3] = c4 * coeffs.l00[axis] - c5 * coeffs.l20[axis];

	return m;
}

__device__ glm::vec3 computeSHIrradiance(const SHCoeffs& coeffs, const glm::vec3& normal)
{
	glm::vec4 n(normal, 1.0f);
	return glm::vec3(glm::dot(n, buildM(coeffs, 0) * n), glm::dot(n, buildM(coeffs, 1) * n), glm::dot(n, buildM(coeffs, 2) * n));
}

__device__ float intersectSphere(const Ray& ray, const glm::vec3& center, float radius, glm::vec3& out_normal)
{
	glm::vec3 oc = ray.getOrigin() - center;
	float a = glm::dot(ray.getDirection(), ray.getDirection());
	float b = 2.0f * glm::dot(oc, ray.getDirection());
	float c = glm::dot(oc, oc) - radius * radius;
	float discriminant = b * b - 4 * a * c;

	if (discriminant < 0.0f)
	{
		return -1.0f;
	}
	else
	{
		auto distance = (-b - glm::sqrt(discriminant)) / (2.0f * a);
		out_normal = glm::normalize(ray.getPoint(distance) - center);
		return distance;
	}
}

__global__ void raytraceSpheresKernel(Camera camera, SHCoeffs sh_coeffs, cudaSurfaceObject_t output, int width, int height)
{
	auto index = util::getThreadIndex2D();
	if (index.x >= width || index.y >= height)
	{
		return;
	}

	float sphere_radius = 0.08f;
	glm::vec3 sphere_centers[] = { {0.0f, 2.0f * sphere_radius + 0.01f, 0.0f},
	{2.0f * sphere_radius + 0.01f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {-2.0f * sphere_radius - 0.01f, 0.0f, 0.0f},
	{4.0f * sphere_radius + 0.02f, -2.0f * sphere_radius - 0.01f, 0.0f}, {2.0f * sphere_radius + 0.01f, -2.0f * sphere_radius - 0.01f, 0.0f},
	{0.0f, -2.0f * sphere_radius - 0.01f, 0.0f}, {-2.0f * sphere_radius - 0.01f, -2.0f * sphere_radius - 0.01f, 0.0f},
	{-4.0f * sphere_radius - 0.02f, -2.0f * sphere_radius - 0.01f, 0.0f} };

	Ray ray(glm::vec3((width - index.x - 0.5f) / width - 0.5f, (height - index.y - 0.5f) / height - 0.5f, -0.25f), glm::vec3(0.0f, 0.0f, 1.0f));

	glm::vec3 color(1.0f);
	glm::vec3 normal;

	int hit_sphere_index = 0;
	float min_distance = FLT_MAX;
	for (int i = 0; i < 9; ++i)
	{
		glm::vec3 temp_normal;
		auto distance = intersectSphere(ray, sphere_centers[i], sphere_radius, temp_normal);
		if (distance > 0.0f && distance < min_distance)
		{
			min_distance = distance;
			hit_sphere_index = i;
			normal = temp_normal;
		}
	}

	if (min_distance != FLT_MAX)
	{
		auto vec3_ptr = reinterpret_cast<glm::vec3*>(&sh_coeffs);
		for (int i = 0; i < 9; ++i)
		{
			if (hit_sphere_index != i)
			{
				vec3_ptr[i] = glm::vec3(0.0f);
			}
		}
		color = glm::vec3(1.0f) * computeSHIrradiance(sh_coeffs, normal) * glm::one_over_pi<float>();
	}

	//BGR is intentionally given.
	color = glm::clamp(color, 0.0f, 1.0f) * 255.0f;
	surf2Dwrite(util::rgbToUint({ color.r, color.g, color.b }), output, index.x * 4, index.y);
}

__global__ void raytracerKernel(Model* model, Camera camera, SHCoeffs sh_coeffs, glm::vec3 light_direction, cudaSurfaceObject_t output, int width, int height, unsigned int* dirty)
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

	glm::vec3 color(1.0f);
	Intersection intersection;

	if (model->intersect(ray, intersection, FLT_MAX))
	{
		auto materials = model->getMaterials().getPtr();
		color = glm::vec3(0.0f);// materials[intersection.material_id].fetchAmbient(intersection.tex_coord);

		//CONSTANT LIGHT
		//color += materials[intersection.material_id].fetchDiffuse(intersection.tex_coord) * 0.9f;

		//DIRECTIONAL LIGHT
		color += materials[intersection.material_id].fetchDiffuse(intersection.tex_coord) * glm::max(0.0f, glm::dot(-light_direction, intersection.normal));

		//ENVIRONMENT LIGHT
		//color += materials[intersection.material_id].fetchDiffuse(intersection.tex_coord) * computeSHIrradiance(sh_coeffs, intersection.normal) * glm::one_over_pi<float>();

		//color += glm::clamp(-glm::transpose(camera.rotation) * intersection.normal, 0.0f, 1.0f);
		//color += glm::abs(glm::transpose(camera.rotation) * intersection.normal);

		if (color.x > 1.0f || color.y > 1.0f || color.z > 1.0f)
		{
			atomicInc(dirty, UINT_MAX);
		}
	}

	//BGR is intentionally given.
	color = glm::clamp(color, 0.0f, 1.0f) * 255.0f;
	surf2Dwrite(util::rgbToUint({ color.r, color.g, color.b }), output, index.x * 4, index.y);
}

bool raytracer(Model* model, const Camera& camera, const SHCoeffs& sh_coeffs, const glm::vec3& light_direction, cudaSurfaceObject_t output, int width, int height)
{
	unsigned int* dirty;
	CHECK_CUDA_ERROR(cudaMalloc(&dirty, sizeof(int)));
	CHECK_CUDA_ERROR(cudaMemset(dirty, 0, sizeof(int)));

	dim3 threads(8, 8);
	dim3 blocks(width / threads.x + 1, height / threads.y + 1);
	raytracerKernel << <blocks, threads >> > (model, camera, sh_coeffs, light_direction, output, width, height, dirty);
	//raytraceSpheresKernel << <blocks, threads >> > (camera, sh_coeffs, output, width, height);

	unsigned int dirty_host;
	CHECK_CUDA_ERROR(cudaMemcpy(&dirty_host, dirty, sizeof(int), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaFree(dirty));

	return dirty_host == 0;
}