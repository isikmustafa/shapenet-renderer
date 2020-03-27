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

SHCoeffs randomSHCoeffs(std::mt19937& generator, std::normal_distribution<float>& distribution)
{
	//First with white light.
	SHCoeffs coeffs;
	coeffs.l00 = glm::vec3(distribution(generator) + 0.5f, distribution(generator) + 0.5f, distribution(generator) + 0.5f);

	coeffs.l1_1 = glm::vec3(distribution(generator), distribution(generator), distribution(generator));
	coeffs.l10 = glm::vec3(distribution(generator), distribution(generator), distribution(generator));
	coeffs.l11 = glm::vec3(distribution(generator), distribution(generator), distribution(generator));

	coeffs.l2_2 = glm::vec3(distribution(generator), distribution(generator), distribution(generator));
	coeffs.l2_1 = glm::vec3(distribution(generator), distribution(generator), distribution(generator));
	coeffs.l20 = glm::vec3(distribution(generator), distribution(generator), distribution(generator));
	coeffs.l21 = glm::vec3(distribution(generator), distribution(generator), distribution(generator));
	coeffs.l22 = glm::vec3(distribution(generator), distribution(generator), distribution(generator));

	return coeffs;
}

int main()
{
	SHCoeffs grace_cathedral;
	grace_cathedral.l00 = { 0.79f, 0.44f, 0.54f };
	grace_cathedral.l1_1 = { 0.39f, 0.35f, 0.60f };
	grace_cathedral.l10 = { -0.34f, -0.18f, -0.27f };
	grace_cathedral.l11 = { -0.29f, -0.06f, 0.01f };
	grace_cathedral.l2_2 = { -0.11f, -0.05f, -0.12f };
	grace_cathedral.l2_1 = { -0.26f, -0.22f, -0.47f };
	grace_cathedral.l20 = { -0.16f, -0.09f, -0.15f };
	grace_cathedral.l21 = { 0.56f, 0.21f, 0.14f };
	grace_cathedral.l22 = { 0.21f, -0.05f, -0.30f };

	SHCoeffs eucalyptus_grove;
	eucalyptus_grove.l00 = { 0.38f, 0.43f, 0.45f };
	eucalyptus_grove.l1_1 = { 0.29f, 0.36f, 0.41f };
	eucalyptus_grove.l10 = { 0.04f, 0.03f, 0.01f };
	eucalyptus_grove.l11 = { -0.10f, -0.10f, -0.09f };
	eucalyptus_grove.l2_2 = { -0.06f, -0.06f, -0.04f };
	eucalyptus_grove.l2_1 = { 0.01f, -0.01f, -0.05f };
	eucalyptus_grove.l20 = { -0.09f, -0.13f, -0.15f };
	eucalyptus_grove.l21 = { -0.06f, -0.05f, -0.04f };
	eucalyptus_grove.l22 = { 0.02f, 0.0f, -0.05f };

	std::ifstream json_file("config.json");
	json model_json;
	json_file >> model_json;
	json_file.close();

	const int screen_width = model_json["imageSideLength"];
	const int screen_height = model_json["imageSideLength"];
	const int number_of_poses = model_json["numberOfPoses"];
	const int number_of_lights = model_json["numberOfLights"];
	const float position_radius = model_json["cameraPositionRadius"];
	Output output(screen_width, screen_height);

	auto fxfy = static_cast<float>(model_json["imageSideLength"]) / 512.0f;
	glm::mat3 intrinsics(glm::vec3(525.0f * fxfy, 0.0f, 0.0f),
		glm::vec3(0.0f, 525.0f * fxfy, 0.0f),
		glm::vec3(256.0f * fxfy, 256.0f * fxfy, 1.0f));

	constexpr glm::vec3 scene_center(0.0f, 0.0f, 0.0f);

	std::vector<std::string> car_ids = model_json["carIds"];
	for (const auto& car_id : car_ids)
	{
		auto rgb_directory_path = std::string(model_json["outputDirectory"]) + car_id + "/rgb/";
		auto pose_directory_path = std::string(model_json["outputDirectory"]) + car_id + "/pose/";
		auto intrinsics_directory_path = std::string(model_json["outputDirectory"]) + car_id + "/intrinsics/";
		auto light_directory_path = std::string(model_json["outputDirectory"]) + car_id + "/light/";

		//Create necessary directories if they don't exist
		if (!std::filesystem::exists(rgb_directory_path))
		{
			std::filesystem::create_directories(rgb_directory_path);
		}
		if (!std::filesystem::exists(pose_directory_path))
		{
			std::filesystem::create_directories(pose_directory_path);
		}
		if (!std::filesystem::exists(intrinsics_directory_path))
		{
			std::filesystem::create_directories(intrinsics_directory_path);
		}
		if (!std::filesystem::exists(light_directory_path))
		{
			std::filesystem::create_directories(light_directory_path);
		}

		auto model_path = std::string(model_json["modelPath"]) + car_id + "/models/";
		std::vector<Model> model;
		model.emplace_back(model_path);

		util::DeviceArray<Model> model_gpu(model);
		std::mt19937 generator((std::random_device())());
		std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
		std::normal_distribution<float> normal_distribution(0.0f, 0.5f);
		int sample_no = 0;
		std::cout << model[0].getBbox().getCenter().x << "," << model[0].getBbox().getCenter().y << "," << model[0].getBbox().getCenter().z << std::endl;
		/*for (int i = 0; i < number_of_poses; ++i)
		{
			auto camera_position = util::sampleSphereUniform(distribution(generator), distribution(generator), position_radius);
			Camera camera(scene_center, camera_position, intrinsics);
			for (int j = 0; j < number_of_lights; ++j)
			{
				//auto disc_pos = util::sampleDiscUniform(distribution(generator), distribution(generator), 1.0f);
				//glm::vec3 light_position(disc_pos.x, 1.5f, disc_pos.y); //y=1.5 plane
				//auto light_direction = glm::normalize(scene_center - light_position);
				glm::vec3 light_direction;
				SHCoeffs sh_coeffs;
				int number_of_samples = 0;
				do
				{
					//light_direction = -util::sampleHemisphereUniform(distribution(generator), distribution(generator), 1.0f);
					sh_coeffs = randomSHCoeffs(generator, normal_distribution);
					++number_of_samples;

				} while (!raytracer(model_gpu.getPtr(), camera, sh_coeffs, light_direction, output.getContent(), screen_width, screen_height));

				auto sample_no_str = std::to_string(sample_no++);
				auto output_name = std::string(6 - sample_no_str.size(), '0').append(sample_no_str);
				output.save(rgb_directory_path + output_name + ".png");
				camera.dumpPoseToFile(pose_directory_path + output_name + ".txt");
				camera.dumpIntrinsicsToFile(intrinsics_directory_path + output_name + ".txt");

				std::ofstream light_file(std::string(light_directory_path + output_name + ".txt"));
				//light_file << light_direction.x << " " << light_direction.y << " " << light_direction.z << std::endl;

				for (auto axis : { 0, 1, 2 })
				{
					light_file << sh_coeffs.l00[axis] << " "
						<< sh_coeffs.l1_1[axis] << " " << sh_coeffs.l10[axis] << " " << sh_coeffs.l11[axis] << " "
						<< sh_coeffs.l2_2[axis] << " " << sh_coeffs.l2_1[axis] << " " << sh_coeffs.l20[axis] << " " << sh_coeffs.l21[axis] << " " << sh_coeffs.l22[axis] << " ";
				}
				light_file << std::endl;

				light_file.close();
			}
		}*/

		constexpr int number_of_frames = 360;
		auto sh_coeffs = eucalyptus_grove;
		auto pos_radius = position_radius;
		auto incr_pos_radius = -0.01f;
		auto theta = 1.0f;
		auto phi = 1.0f;
		for (int i = 0; i < number_of_frames; ++i)
		{
			auto camera_position = util::sampleSphereUniform(0.25f, static_cast<float>(i) / number_of_frames, pos_radius);
			Camera camera(scene_center, glm::vec3(camera_position.x, camera_position.z, camera_position.y), intrinsics);
			auto light_direction = -util::sampleHemisphereUniform(theta, glm::abs(phi), 1.0f);

			raytracer(model_gpu.getPtr(), camera, sh_coeffs, light_direction, output.getContent(), screen_width, screen_height);

			auto sample_no_str = std::to_string(sample_no++);
			auto output_name = std::string(6 - sample_no_str.size(), '0').append(sample_no_str);
			output.save(rgb_directory_path + output_name + ".png");
			camera.dumpPoseToFile(pose_directory_path + output_name + ".txt");
			camera.dumpIntrinsicsToFile(intrinsics_directory_path + output_name + ".txt");

			std::ofstream light_file(std::string(light_directory_path + output_name + ".txt"));
			light_file << light_direction.x << " " << light_direction.y << " " << light_direction.z << std::endl;
			light_file.close();

			pos_radius += incr_pos_radius;
			if (pos_radius > 1.6f)
			{
				pos_radius = 1.6f;
				incr_pos_radius = -incr_pos_radius;
			}
			else if (pos_radius < 1.0f)
			{
				pos_radius = 1.0f;
				incr_pos_radius = -incr_pos_radius;
			}

			theta -= 1.0f / 360.0f;
			phi -= 1.0f / 180.0f;
		}

		std::cout << car_id << " is finished" << std::endl;
	}

	//Fixed pose and rotating directional light.
	/*auto camera_position = glm::normalize(glm::vec3(0.4f, 0.4f, -1.0f)) * position_radius;
	Camera camera(scene_center, camera_position, intrinsics);
	constexpr int number_of_frames = 360;
	for (int i = 0; i < number_of_frames; ++i)
	{
		auto disc_pos = util::sampleDiscUniform(1.0f, static_cast<float>(i + 1) / number_of_frames, 1.0f);
		glm::vec3 light_position(disc_pos.x, 1.5f, disc_pos.y); //y=1.5 plane
		auto light_direction = glm::normalize(scene_center - light_position);

		raytracer(model_gpu.getPtr(), camera, light_direction, output.getContent(), screen_width, screen_height);

		auto sample_no_str = std::to_string(sample_no++);
		auto output_name = std::string(6 - sample_no_str.size(), '0').append(sample_no_str);
		output.save(rgb_directory_path + output_name + ".png");
		camera.dumpPoseToFile(pose_directory_path + output_name + ".txt");
		camera.dumpIntrinsicsToFile(intrinsics_directory_path + output_name + ".txt");

		std::ofstream light_file(std::string(light_directory_path + output_name + ".txt");
		light_file << light_direction.x << " " << light_direction.y << " " << light_direction.z << std::endl;
		light_file.close();
	}*/

	//Fixed SH light and rotating camera.
	/*constexpr int number_of_frames = 360;
	auto sh_coeffs = eucalyptus_grove;
	for (int i = 0; i < number_of_frames; ++i)
	{
		auto camera_position = util::sampleSphereUniform(0.25f, static_cast<float>(i) / number_of_frames, position_radius);
		Camera camera(scene_center, glm::vec3(camera_position.x, camera_position.z, camera_position.y), intrinsics);

		raytracer(model_gpu.getPtr(), camera, sh_coeffs, {}, output.getContent(), screen_width, screen_height);

		auto sample_no_str = std::to_string(sample_no++);
		auto output_name = std::string(6 - sample_no_str.size(), '0').append(sample_no_str);
		output.save(rgb_directory_path + output_name + ".png");
		camera.dumpPoseToFile(pose_directory_path + output_name + ".txt");
		camera.dumpIntrinsicsToFile(intrinsics_directory_path + output_name + ".txt");

		std::ofstream light_file(std::string(light_directory_path + output_name + ".txt"));
		for (auto axis : { 0, 1, 2 })
		{
			light_file << sh_coeffs.l00[axis] << " "
				<< sh_coeffs.l1_1[axis] << " " << sh_coeffs.l10[axis] << " " << sh_coeffs.l11[axis] << " "
				<< sh_coeffs.l2_2[axis] << " " << sh_coeffs.l2_1[axis] << " " << sh_coeffs.l20[axis] << " " << sh_coeffs.l21[axis] << " " << sh_coeffs.l22[axis] << " ";
		}
		light_file << std::endl;
		light_file.close();
	}*/

	//Fixed light and rotating camera.
	/*constexpr int number_of_frames = 360;
	auto disc_pos = util::sampleDiscUniform(1.0f, 0.0f, 1.0f);
	glm::vec3 light_position(disc_pos.x, 1.5f, disc_pos.y); //y=1.5 plane
	auto light_direction = glm::normalize(scene_center - light_position);
	for (int i = 0; i < number_of_frames; ++i)
	{
		auto camera_position = util::sampleSphereUniform(0.5f, static_cast<float>(i) / number_of_frames, position_radius);
		Camera camera(scene_center, glm::vec3(camera_position.x, camera_position.z, camera_position.y), intrinsics);

		raytracer(model_gpu.getPtr(), camera, light_direction, output.getContent(), screen_width, screen_height);

		auto sample_no_str = std::to_string(sample_no++);
		auto output_name = std::string(6 - sample_no_str.size(), '0').append(sample_no_str);
		output.save(rgb_directory_path + output_name + ".png");
		camera.dumpPoseToFile(pose_directory_path + output_name + ".txt");
		camera.dumpIntrinsicsToFile(intrinsics_directory_path + output_name + ".txt");

		std::ofstream light_file(std::string(light_directory_path + output_name + ".txt");
		light_file << light_direction.x << " " << light_direction.y << " " << light_direction.z << std::endl;
		light_file.close();
	}*/

	/*//Fixed pose and changing SH coefficients.
	SHCoeffs sh_coeffs;
	constexpr int number_of_frames = 360;
	//auto camera_position = glm::normalize(glm::vec3(0.4f, 0.4f, -1.0f)) * position_radius;
	//Camera camera(scene_center, camera_position, intrinsics);
	Camera camera(scene_center, glm::vec3(0.0f, 0.0f, -1.0f), intrinsics);

	auto vec3_ptr = reinterpret_cast<glm::vec3*>(&sh_coeffs);
	float ambient_vals[] = { 0.5f, 1.0f, 0.0f, 0.5f };
	float other_vals[] = { 0.0f, -1.0f, 1.0f, 0.0f };
	float frames[] = { 30.0f, 60.0f, 30.0f };
	for (int coeff_no = 0; coeff_no < 9; ++coeff_no)
	{
		for (int next_step_idx = 1; next_step_idx < 4; ++next_step_idx)
		{
			auto start = coeff_no == 0 ? ambient_vals[next_step_idx - 1] : other_vals[next_step_idx - 1];
			auto end = coeff_no == 0 ? ambient_vals[next_step_idx] : other_vals[next_step_idx];
			auto increment = (end - start) / frames[next_step_idx - 1];

			for (auto coeff = start; (coeff < end && start < end) || (coeff > end && end < start); coeff += increment)
			{
				vec3_ptr[coeff_no] = glm::vec3(coeff);
				raytracer(model_gpu.getPtr(), camera, sh_coeffs, {}, output.getContent(), screen_width, screen_height);

				auto sample_no_str = std::to_string(sample_no++);
				auto output_name = std::string(6 - sample_no_str.size(), '0').append(sample_no_str);
				output.save(rgb_directory_path + output_name + ".png");
				camera.dumpPoseToFile(pose_directory_path + output_name + ".txt");
				camera.dumpIntrinsicsToFile(intrinsics_directory_path + output_name + ".txt");

				std::ofstream light_file(std::string(light_directory_path + output_name + ".txt"));
				for (auto axis : { 0, 1, 2 })
				{
					light_file << sh_coeffs.l00[axis] << " "
						<< sh_coeffs.l1_1[axis] << " " << sh_coeffs.l10[axis] << " " << sh_coeffs.l11[axis] << " "
						<< sh_coeffs.l2_2[axis] << " " << sh_coeffs.l2_1[axis] << " " << sh_coeffs.l20[axis] << " " << sh_coeffs.l21[axis] << " " << sh_coeffs.l22[axis] << " ";
				}
				light_file << std::endl;
				light_file.close();
			}
		}
	}*/

	/*
	//Hemispherical moving light and moving camera doing zoom-in zoom-out
	constexpr int number_of_frames = 360;
	auto sh_coeffs = eucalyptus_grove;
	auto pos_radius = position_radius;
	auto incr_pos_radius = -0.01f;
	auto theta = 1.0f;
	auto phi = 1.0f;
	for (int i = 0; i < number_of_frames; ++i)
	{
		auto camera_position = util::sampleSphereUniform(0.25f, static_cast<float>(i) / number_of_frames, pos_radius);
		Camera camera(scene_center, glm::vec3(camera_position.x, camera_position.z, camera_position.y), intrinsics);
		auto light_direction = -util::sampleHemisphereUniform(theta, glm::abs(phi), 1.0f);
		auto tmp = light_direction.z;
		light_direction.z = light_direction.y;
		light_direction.y = tmp;

		raytracer(model_gpu.getPtr(), camera, sh_coeffs, light_direction, output.getContent(), screen_width, screen_height);

		auto sample_no_str = std::to_string(sample_no++);
		auto output_name = std::string(6 - sample_no_str.size(), '0').append(sample_no_str);
		output.save(rgb_directory_path + output_name + ".png");
		camera.dumpPoseToFile(pose_directory_path + output_name + ".txt");
		camera.dumpIntrinsicsToFile(intrinsics_directory_path + output_name + ".txt");

		std::ofstream light_file(std::string(light_directory_path + output_name + ".txt"));
		light_file << light_direction.x << " " << light_direction.y << " " << light_direction.z << std::endl;
		light_file.close();

		pos_radius += incr_pos_radius;
		if (pos_radius > 1.6f)
		{
			pos_radius = 1.6f;
			incr_pos_radius = -incr_pos_radius;
		}
		else if (pos_radius < 1.0f)
		{
			pos_radius = 1.0f;
			incr_pos_radius = -incr_pos_radius;
		}

		theta -= 1.0f / 360.0f;
		phi -= 1.0f / 180.0f;
	}
	*/

	return 0;
}
