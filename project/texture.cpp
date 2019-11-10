#include "texture.h"
#include "util.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

Texture::Texture(const std::string& filename)
{
	if (filename.empty())
	{
		return;
	}

	int channel;
	int width;
	int height;
	auto data = stbi_load(filename.c_str(), &width, &height, &channel, STBI_rgb_alpha);

	std::cout << channel << std::endl;
	std::cout << width << std::endl;
	std::cout << height << std::endl;

	if (!data)
	{
		throw std::runtime_error("Error: Image cannot be loaded");
	}

	//Create the texture on device.
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned); //RGBA

	CHECK_CUDA_ERROR(cudaMallocArray(&m_cuda_array, &channel_desc, width, height));
	CHECK_CUDA_ERROR(cudaMemcpyToArray(m_cuda_array, 0, 0, data, width * height * 4, cudaMemcpyHostToDevice));

	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = m_cuda_array;

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(tex_desc));
	tex_desc.addressMode[0] = cudaTextureAddressMode(cudaAddressModeWrap);
	tex_desc.addressMode[1] = cudaTextureAddressMode(cudaAddressModeWrap);
	tex_desc.filterMode = cudaTextureFilterMode(cudaFilterModeLinear);
	tex_desc.readMode = cudaReadModeNormalizedFloat;
	tex_desc.normalizedCoords = 1;

	CHECK_CUDA_ERROR(cudaCreateTextureObject(&m_texture, &res_desc, &tex_desc, nullptr));

	stbi_image_free(data);
}

Texture::Texture(Texture&& rhs)
{
	std::swap(m_cuda_array, rhs.m_cuda_array);
	std::swap(m_texture, rhs.m_texture);
}

Texture& Texture::operator=(Texture&& rhs)
{
	std::swap(m_cuda_array, rhs.m_cuda_array);
	std::swap(m_texture, rhs.m_texture);

	return *this;
}

Texture::~Texture()
{
	CHECK_CUDA_ERROR(cudaDestroyTextureObject(m_texture));
	CHECK_CUDA_ERROR(cudaFreeArray(m_cuda_array));
}
