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

	constexpr int number_of_poses = 1;
	constexpr int number_of_lights = 100;
	constexpr float position_radius = 1.3f;

	std::mt19937 generator((std::random_device())());
	std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
	int sample_no = 0;
	for (int i = 0; i < number_of_poses; ++i)
	{
		auto camera_position = util::sampleSphereUniform(distribution(generator), distribution(generator), position_radius);
		Camera camera(glm::vec3(0.0f), camera_position);
		for (int j = 0; j < number_of_lights; ++j)
		{
			auto disc_pos = util::sampleDiscUniform(distribution(generator), distribution(generator), 1.0f);
			glm::vec3 light_position(disc_pos.x, 1.5f, disc_pos.y); //y=1 plane
			auto light_direction = -light_position;

			raytracer(model_gpu.getPtr(), camera, light_direction, output.getContent(), screen_width, screen_height);

			auto sample_no_str = std::to_string(sample_no++);
			auto output_name = std::string(6 - sample_no_str.size(), '0').append(sample_no_str);
			output.save("C:/Users/Mustafa/Desktop/custon_train/1a0bc9ab92c915167ae33d942430658c/rgb/" + output_name + ".png");
			camera.dumpToFile("C:/Users/Mustafa/Desktop/custon_train/1a0bc9ab92c915167ae33d942430658c/pose/" + output_name + ".txt");
		}
	}

	return 0;
}
