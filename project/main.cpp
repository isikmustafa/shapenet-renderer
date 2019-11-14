#include "model.h"
#include "output.h"
#include "device_array.h"
#include "raytracer.h"
#include "camera.h"

#include <fstream>
#include <iostream>
#include <random>
#include <nlohmann/json.hpp>
#include <filesystem>

using nlohmann::json;

int main()
{
	std::ifstream json_file("config.json");
	json model_json;
	json_file >> model_json;
	json_file.close();

	//Create necessary directories if they don't exist
	if (!std::filesystem::exists(std::string(model_json["rgbDirectoryPath"])))
	{
		std::filesystem::create_directories(std::string(model_json["rgbDirectoryPath"]));
	}
	if (!std::filesystem::exists(std::string(model_json["poseDirectoryPath"])))
	{
		std::filesystem::create_directories(std::string(model_json["poseDirectoryPath"]));
	}
	if (!std::filesystem::exists(std::string(model_json["intrinsicsDirectoryPath"])))
	{
		std::filesystem::create_directories(std::string(model_json["intrinsicsDirectoryPath"]));
	}

	const int screen_width = model_json["imageSideLength"];
	const int screen_height = model_json["imageSideLength"];
	Output output(screen_width, screen_height);
	std::vector<Model> model;
	model.emplace_back(model_json["singleModelPath"]);
	
	util::DeviceArray<Model> model_gpu(model);

	const int number_of_poses = model_json["numberOfPoses"];
	const int number_of_lights = model_json["numberOfLights"];
	const float position_radius = model_json["cameraPositionRadius"];

	auto fxfy = model_json["imageSideLength"] / 512.0f;
	glm::mat3 intrinsics(glm::vec3(525.0f * fxfy, 0.0f, 0.0f),
		glm::vec3(0.0f, 525.0f * fxfy, 0.0f),
		glm::vec3(256.0f * fxfy, 256.0f * fxfy, 1.0f));

	std::mt19937 generator((std::random_device())());
	std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
	int sample_no = 0;
	for (int i = 0; i < number_of_poses; ++i)
	{
		auto camera_position = util::sampleSphereUniform(distribution(generator), distribution(generator), position_radius);
		Camera camera(glm::vec3(0.0f), camera_position, intrinsics);
		for (int j = 0; j < number_of_lights; ++j)
		{
			auto disc_pos = util::sampleDiscUniform(distribution(generator), distribution(generator), 1.0f);
			glm::vec3 light_position(disc_pos.x, 1.5f, disc_pos.y); //y=1.5 plane
			auto light_direction = -light_position;

			raytracer(model_gpu.getPtr(), camera, light_direction, output.getContent(), screen_width, screen_height);

			auto sample_no_str = std::to_string(sample_no++);
			auto output_name = std::string(6 - sample_no_str.size(), '0').append(sample_no_str);
			output.save(std::string(model_json["rgbDirectoryPath"]) + output_name + ".png");
			camera.dumpPoseToFile(std::string(model_json["poseDirectoryPath"]) + output_name + ".txt");
			camera.dumpIntrinsicsToFile(std::string(model_json["intrinsicsDirectoryPath"]) + output_name + ".txt");
		}
	}

	return 0;
}
