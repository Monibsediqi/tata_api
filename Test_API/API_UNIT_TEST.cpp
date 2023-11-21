// Test_API.cpp : This file contains the 'main' function. 
// Program execution begins and ends in this file.

// system include
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <tuple>
#include <map>
#include <mutex>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <random>
#include <map>
#include <unordered_map>

// library include
#include "TataBase.h"


using namespace std;

#define PRINT(x) std::cout << x << std::endl;
#define INFO(x) std::cout << "INFO: " << x << std::endl;
#define ERR(x) std::cerr << "ERROR: " << x << std::endl;

vector<int16_t> convertFloatToInt16(const vector<float> vec_data) {
	vector<int16_t> vec_int16_data;
	for (auto x : vec_data) {
		vec_int16_data.push_back((int16_t)x);
	}
	return vec_int16_data;
}
vector<int16_t> convertShortToInt16(const vector<short> vec_data) {
	vector<int16_t> vec_int16_data;
	for (auto x : vec_data) {
		vec_int16_data.push_back((int16_t)x);
	}
	return vec_int16_data;
}
vector<vector<vector<int16_t>>> convertFloatToInt16(const vector<vector<vector<float>>> v_input){
	std::vector<std::vector<std::vector<int16_t>>> v_output; // Output vector of int16_t

	// Iterate over each element of the input vector
	for (const auto& vec1 : v_input) {
		std::vector<std::vector<int16_t>> temp_vec1;  // Temporary vector to store intermediate results

		// Iterate over each element of the inner vector
		for (const auto& vec2 : vec1) {
			std::vector<int16_t> temp_vec2;  // Temporary vector to store intermediate results

			// Iterate over each element of the innermost vector and convert to int16_t
			for (const auto& element : vec2) {
				temp_vec2.push_back(static_cast<int16_t>(element));
			}

			temp_vec1.push_back(temp_vec2);
		}

		v_output.push_back(temp_vec1);
	}
}

vector<short> readRawFile1D(string& raw_file_path, int d, int w, int h) {
	std::ifstream input_file(raw_file_path, std::ios::in | std::ios::binary);

	std::vector<short> v_pixel_val;
	if (!input_file) {
		ERR("Error: Failed to open file. " << raw_file_path);
		ERR("Returning an empty vector. ");
		return v_pixel_val; //return empty vector (nothing)
	};

	if (input_file.is_open()) {
		PRINT("Openning File successful. ");
		int16_t data; // size of the dicom data is 16 bit (2bytes)

		int count = 0;
		int img_resolution = d * w * h;
		while (input_file.read(/*buffer*/(char*)&data, /*buffer size*/sizeof(data))) {

			v_pixel_val.push_back((short)(data));
			++count;
			if (count > img_resolution) {
				throw std::invalid_argument(std::string("Wrong resolution value. Image size is larger than ") +
					std::to_string(d) + " x " + std::to_string(w) + "x" + std::to_string(h));
			}
		};
		input_file.close();
	}
	PRINT("Reading file successful.");
	return v_pixel_val;
};
void writeRawData1D(vector<int16_t> data, string filename) {

	INFO("checking 200 pixels value: ");
	int counter = 0;
	for (auto x : data) {
		std::cout << x << " ";
		counter++;
		if (counter == 50) {
			break;
		}
	}
	auto file = std::ofstream(filename, std::ios::out | std::ios::binary);
	if (file.is_open()) {
		file.write((char*)&data[0], sizeof(int16_t) * data.size());
		file.close();
	}
}

void testAPI() {

	string LUNG_VOL_RAW = "F:\\2023\\AI\\sample_data\\tisepx_a\\lung_vol_raw\\JB0006_CXR_0base_201229.raw";
	
		int64_t XRAY_D = 1;
		int64_t XRAY_W = 2652;
		int64_t XRAY_H = 2450;
	
		vector<short> vec_img_A = readRawFile1D(LUNG_VOL_RAW, XRAY_D, XRAY_W, XRAY_H); // read image A
		//vector<float_t> vec_img_A_float(vec_img_A.begin(), vec_img_A.end());
		tata::PatientInfo p_info;
		p_info.age = 74;
		p_info.sex = "M";


		tata::PatientData p_data;

		p_data.img_original = vec_img_A;
		p_data.img_size_x = XRAY_W;
		p_data.img_size_y = XRAY_H;
		p_data.img_size_z = XRAY_D;
		p_data.img_spacing_x = 0.1430;
		p_data.img_spacing_y = 0.1430;
		p_data.img_spacing_z = 1.0;

		const string	weight_path = "F:\\2023\\AI\\App\\scripted_models\\";
		const string	xray2lung_weight = "xray2lung.mipx";
		const string	lung_reg_weight = "lungregression.mipx";
		const string 	lung2covid_weight = "lung2covid.mipx";
		const string 	xray2heart_weight = "xray2heart.mipx";
		const string    lung2vessel = "lung2vessel.mipx";
		const string    xray2bone = "xray2bone.mipx";

		tata::Configuration model_config;
		model_config.weight_path = weight_path;
		model_config.lung_segmentation.weight_name = xray2lung_weight;
		model_config.lung_regression.weight_name = lung_reg_weight;
		model_config.COVID.weight_name = lung2covid_weight;
		model_config.heart_segmentation.weight_name = xray2heart_weight;
		model_config.vessel_segmentation.weight_name = lung2vessel;
		model_config.bone_segmentation.weight_name = xray2bone;

		vector<tata::AnalysisPlan> analysis_plan;
		analysis_plan.push_back(tata::AnalysisPlan::CXR_LUNG);
		analysis_plan.push_back(tata::AnalysisPlan::CXR_BONE);
		analysis_plan.push_back(tata::AnalysisPlan::CXR_HEART);
		analysis_plan.push_back(tata::AnalysisPlan::CXR_VASCULAR);
		analysis_plan.push_back(tata::AnalysisPlan::CXR_COVID);


		// -- API base setup --
		tata::TataBase tata_api (model_config);

		//Setup TiSepX Analysis Plan
		tata::Analysis tisep_analysis(model_config);
		tisep_analysis.setPatientInfo(p_info);
		tisep_analysis.setPatientData(p_data);
		tisep_analysis.setAnalysisPlan(analysis_plan);

		tisep_analysis.doAnalysis();
		PRINT("Analysis completed!");
		
		unordered_map<string, tata::Layer> out_layers = tisep_analysis.getOutputLayers();

		//Lung
		vector<int16_t> v_int_data = convertShortToInt16(out_layers["lung"].v_tisepx_image);
		writeRawData1D(v_int_data, "F:\\2023\\AI\\sample_data\\tisepx_b\\lung_seg_raw\\JB0006_CXR_0base_201229.raw");

		//Bone
		v_int_data = convertShortToInt16(out_layers["bone"].v_tisepx_image);
		writeRawData1D(v_int_data, "F:\\2023\\AI\\sample_data\\tisepx_b\\bone_seg_raw\\JB0006_CXR_0base_201229.raw");

		//Heart
		v_int_data = convertShortToInt16(out_layers["heart"].v_tisepx_image);
		writeRawData1D(v_int_data, "F:\\2023\\AI\\sample_data\\tisepx_b\\heart_seg_raw\\JB0006_CXR_0base_201229.raw");

		//Vessel
		v_int_data = convertShortToInt16(out_layers["vascular"].v_tisepx_image);
		writeRawData1D(v_int_data, "F:\\2023\\AI\\sample_data\\tisepx_b\\vessel_seg_raw\\JB0006_CXR_0base_201229.raw");

		//COVID
		v_int_data = convertShortToInt16(out_layers["covid"].v_tisepx_image);
		writeRawData1D(v_int_data, "F:\\2023\\AI\\sample_data\\tisepx_b\\covid_seg_raw\\JB0006_CXR_0base_201229.raw");

}



int main()
{
	testAPI();
	INFO("Test completed!")
}