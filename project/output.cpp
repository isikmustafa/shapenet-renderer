#include "output.h"
#include "util.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

Output::Output(int width, int height)
	: m_width(width)
	, m_height(height)
{
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	CHECK_CUDA_ERROR(cudaMallocArray(&m_content_array, &channel_desc, width, height, cudaArraySurfaceLoadStore));

	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;

	res_desc.res.array.array = m_content_array;
	CHECK_CUDA_ERROR(cudaCreateSurfaceObject(&m_content, &res_desc));
}

Output::Output(Output&& rhs)
	: m_width(rhs.m_width)
	, m_height(rhs.m_height)
	, m_content_array(rhs.m_content_array)
	, m_content(rhs.m_content)
{
	rhs.m_width = 0;
	rhs.m_height = 0;
	rhs.m_content_array = nullptr;
	rhs.m_content = 0;
}

Output& Output::operator=(Output&& rhs)
{
	m_width = rhs.m_width;
	m_height = rhs.m_height;
	m_content_array = rhs.m_content_array;
	m_content = rhs.m_content;
	rhs.m_width = 0;
	rhs.m_height = 0;
	rhs.m_content_array = nullptr;
	rhs.m_content = 0;

	return *this;
}

Output::~Output()
{
	CHECK_CUDA_ERROR(cudaDestroySurfaceObject(m_content));
	CHECK_CUDA_ERROR(cudaFreeArray(m_content_array));
}

void Output::save(const std::string& filename)
{
	std::unique_ptr<unsigned char[]> data(new unsigned char[m_width * m_height * 4]);

	CHECK_CUDA_ERROR(cudaMemcpyFromArray(data.get(), m_content_array, 0, 0, m_width * m_height * 4, cudaMemcpyDeviceToHost));

	auto image_format = filename.substr(filename.find_last_of('.') + 1);
	if (image_format == "png")
	{
		stbi_write_png(filename.c_str(), m_width, m_height, 4, data.get(), 4 * m_width);
	}
	else if (image_format == "bmp")
	{
		stbi_write_bmp(filename.c_str(), m_width, m_height, 4, data.get());
	}
	else if (image_format == "tga")
	{
		stbi_write_tga(filename.c_str(), m_width, m_height, 4, data.get());
	}
}
