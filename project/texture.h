#pragma once

#include <string>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

class Texture
{
public:
	Texture(const std::string& filename);
	Texture(const Texture&) = delete;
	Texture(Texture&&);
	Texture& operator=(const Texture&) = delete;
	Texture& operator=(Texture&&);
	~Texture();

#ifdef __NVCC__
	__device__ glm::vec3 fetch(const glm::vec2& tex_coord) const
	{
		if (!m_cuda_array)
		{
			return glm::vec3(1.0f);
		}

		float4 color = tex2D<float4>(m_texture, tex_coord.x, tex_coord.y);
		return glm::vec3(color.z, color.y, color.x); //BGR TO RGB
	}
#endif

private:
	cudaArray_t m_cuda_array{ nullptr };
	cudaTextureObject_t m_texture{ 0 };
};