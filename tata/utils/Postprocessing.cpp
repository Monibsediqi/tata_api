/*
Author: Monib Sediqi @ Medical IP Inc.
Email: monib.sediqi@medicalip.com | kh.monib@gmail.com
Date: 2023-02-01
All rights reserved.
*/

#include "Postprocessing.h"
#define PRINT(x) std::cout<< x <<std::endl;
#define INFO(x) std::cout<<"INFO: " <<x<<std::endl;

///////////////////////////// Global Functions /////////////////////////////////

////void writeRawData3D(vector<vector<vector<int16_t>>> v_data, string filename) {
//	auto outfile = ofstream(filename, ios::out | ios::binary);
//	if (outfile.is_open()) {
//		//loop through each pixel in the 3D vector and write it to the output file
//		for (int i = 0; i < v_data.size(); i++) {
//			for (int j = 0; j < v_data[i].size(); j++) {
//				for (int k = 0; k < v_data[i][j].size(); k++) {
//					//print("Pixel value before writing: " << reinterpret_cast<char*> (v_data[i][j][k]))
//					outfile.write((char*)&v_data[i][j][k], sizeof(short));
//
//				}
//			}
//		}
//	}
//	INFO("file sved to: "<<filename);
//}


vector<int16_t> convertFloatToInt16(const vector<float_t> vec_data) {
	vector<int16_t> vec_int16_data;
	for (auto x : vec_data) {
		vec_int16_data.push_back((int16_t)x);
	}
	return vec_int16_data;
}
std::vector<float_t> convertTensorToVector(const torch::Tensor& tensor) { // Converts kCPU tensor
	std::vector<float_t> vec_data(tensor.data_ptr<float_t>(), tensor.data_ptr<float_t>() + tensor.numel());
	return vec_data;
}
vector<vector<vector<float_t>>> convertTensorTo3DVector(const torch::Tensor& tensor) {
	cout << "converting to vector<float_t>" << endl;
	vector<vector<vector<float_t>>> vec_data(tensor.size(0), vector <vector<float_t>>(tensor.size(1), vector<float_t>(tensor.size(2))));
	for (int i = 0; i < tensor.size(0); i++) {
		for (int j = 0; j < tensor.size(1); j++) {
			for (int k = 0; k < tensor.size(2); k++) {
				vec_data[i][j][k] = (tensor[i][j][k].item<float_t>());
			}
		}
	}
	cout << "conversion successful!";
	return vec_data;
}
vector<vector<vector<int16_t>>> convertTensorTo3DVectorInt16(const torch::Tensor& tensor) {
	cout << "converting to vector<int16_t>" << endl;
	vector<vector<vector<int16_t>>> vec_data(tensor.size(0), vector <vector<int16_t>>(tensor.size(1), vector<int16_t>(tensor.size(2))));
	int16_t min = 1000000;
	int16_t max = -1000000;
	for (int i = 0; i < tensor.size(0); i++) {
		cout << "i:" << i;
		for (int j = 0; j < tensor.size(1); j++) {
			for (int k = 0; k < tensor.size(2); k++) {
				if ((int16_t) tensor[i][j][k].item<float_t>() < min) {
					min = (int16_t)tensor[i][j][k].item<float_t>();
				}
				if ((int16_t) tensor[i][j][k].item<float_t>() > max) 
					max = (int16_t) tensor[i][j][k].item<float_t>(); {
				}
				vec_data[i][j][k] = (int16_t)(tensor[i][j][k].item<float_t>());
			}
		}
	}
	
	INFO("converstion successful!\n");
	return vec_data;
}

// -------------------------------- LUNG VOLUMETRY --------------------------------
//Tensor getBiggestConnectedRegion(const cv::Mat& gen_lung, int n_region = 2) {
//	cv::Mat grayscale;
//	cv::cvtColor(gen_lung, grayscale, cv::COLOR_BGR2GRAY);
//	cv::Mat binary;
//	cv::threshold(grayscale, binary, 0, 255, cv::THRESH_BINARY);
//
//	cv::Mat labels;
//	int n_connected_region = cv::connectedComponents(binary, labels);
//
//	vector<int> n_connected_region_counts(n_connected_region, 0);
//	for (int i = 0; i < labels.rows; ++i) {
//		for (int j = 0; j < labels.cols; ++j) {
//			int label = labels.at<int>(i, j);
//			n_connected_region_counts[label]++;
//		}
//	}
//
//	n_connected_region_counts[0] = 0;
//
//	vector<int> biggest_regions_index;
//	for (int i = 1; i <= n_region && i < n_connected_region_counts.size(); ++i) {
//		int max_index = std::distance(n_connected_region_counts.begin(), std::max_element(n_connected_region_counts.begin(), n_connected_region_counts.end()));
//		biggest_regions_index.push_back(max_index);
//		n_connected_region_counts[max_index] = 0;
//	}
//
//	cv::Mat result = cv::Mat::zeros(gen_lung.size(), CV_8UC1);
//
//	for (int i = 0; i < gen_lung.rows; ++i) {
//		for (int j = 0; j < gen_lung.cols; ++j) {
//			int label = labels.at<int>(i, j);
//			if (std::find(biggest_regions_index.begin(), biggest_regions_index.end(), label) != biggest_regions_index.end()) {
//				result.at<uchar>(i, j) = 255;
//			}
//		}
//	}
//
//	// Convert OpenCV Mat to Torch Tensor
//	cv::Mat result_normalized;
//	result.convertTo(result_normalized, CV_32F, 1.0 / 255.0);
//	torch::Tensor tensor_result = torch::from_blob(result_normalized.data, { 1, result_normalized.rows, result_normalized.cols }, torch::kFloat32).clone();
//
//	return tensor_result;
//}
//cv::Mat tensorToImage(const torch::Tensor& tensor) {
//	// Convert Torch Tensor to OpenCV Mat
//	torch::Tensor tensor_normalized = tensor * 255.0;
//	tensor_normalized = tensor_normalized.to(torch::kU8);
//	cv::Mat result(tensor_normalized.size(1), tensor_normalized.size(2), CV_8UC1, tensor_normalized.data_ptr<uint8_t>());
//
//	return result.clone();
//}