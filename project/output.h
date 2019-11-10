#pragma once

#include <string>
#include <cuda_runtime.h>

class Output
{
public:
	Output(int width, int height);
	Output(const Output&) = delete;
	Output(Output&&);
	Output& operator=(const Output&) = delete;
	Output& operator=(Output&&);
	~Output();

	void save(const std::string& filename);
	cudaSurfaceObject_t getContent() const { return m_content; }
	
private:
	int m_width;
	int m_height;
	cudaArray_t m_content_array{ nullptr };
	cudaSurfaceObject_t m_content{ 0 };
};