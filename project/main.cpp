#include "model.h"
#include "output.h"
#include "device_array.h"
#include "raytracer.h"
#include "camera.h"

#include <iostream>

int main()
{
	const std::string path = "C:/Users/Mustafa/Downloads/ShapeNetCore.v2/02958343/1a1dcd236a1e6133860800e6696b8284/models/";
	constexpr int screen_width = 512;
	constexpr int screen_height = 512;
	Output output(screen_width, screen_height);
	std::vector<Model> model;
	model.emplace_back(path);
	Camera camera(glm::vec3(0.0), glm::vec3(0.0f, 0.0f, -1.3f));
	
	util::DeviceArray<Model> model_gpu(model);

	raytracer(model_gpu.getPtr(), camera, output.getContent(), screen_width, screen_height);

	output.save("C:/Users/Mustafa/Desktop/osman.png");

	return 0;
}
