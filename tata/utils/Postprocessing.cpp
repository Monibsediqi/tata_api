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

