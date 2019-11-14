#include "model.h"
#include "output.h"
#include "device_array.h"
#include "raytracer.h"
#include "camera.h"

#include <iostream>
#include <random>

int main()
{
	const std::string path = "C:/Users/Mustafa/Downloads/ShapeNetCore.v2/02958343/1a0bc9ab92c915167ae33d942430658c/models/";
	constexpr int screen_width = 512;
	constexpr int screen_height = 512;
	Output output(screen_width, screen_height);
	std::vector<Model> model;
	model.emplace_back(path);
	
	util::DeviceArray<Model> model_gpu(model);

	constexpr int number_of_poses = 50;
	constexpr int number_of_light = 30;
	constexpr float position_radius = 1.3f;

	std::mt19937 generator((std::random_device())());
	std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
	for (int i = 0; i < number_of_poses; ++i)
	{
		auto u1 = distribution(generator);
		auto u2 = distribution(generator);
		auto position = util::sampleSphereUniform(u1, u2, position_radius);

		auto model_center = model[0].getBbox().getCenter();
		Camera camera(model_center, position + model_center);
		raytracer(model_gpu.getPtr(), camera, output.getContent(), screen_width, screen_height);

		auto i_str = std::to_string(i);
		auto output_name = std::string(6 - i_str.size(), '0').append(i_str);
		output.save("C:/Users/Mustafa/Desktop/1a0bc9ab92c915167ae33d942430658c/rgb/" + output_name + ".png");
		camera.dumpToFile("C:/Users/Mustafa/Desktop/1a0bc9ab92c915167ae33d942430658c/pose/" + output_name + ".txt");
	}

	return 0;
}
