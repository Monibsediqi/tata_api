/*
Author: Monib Sediqi @ Medical IP Inc.
Email: monib.sediqi@medicalip.com | kh.monib@gmail.com
Date: 2023-02-01
All rights reserved.
*/

#include "Preprocessing.h"

using namespace std;
using namespace torch;
namespace F = torch::nn::functional;

#define PRINT(x) std::cout << x << std::endl;
#define INFO(x) std::cout << "INFO: " << x << std::endl;
#define ERR(x) std::cerr << "ERROR: " << x << std::endl;

////////////////////////////////////// Global Functions ///////////////////////////////////

// ----------------------------- VECTOR OPERATIONS ------------------------------------

float_t vecAvg(std::vector<float_t>& v) {
	float sum = 0.0;
	for (float_t x : v) {
		sum += x;
	}
	return sum / v.size();
};
float_t vecStd(std::vector<float_t>& v) {
	float sum = 0.0;
	float avg = vecAvg(v);
	for (float_t x : v) {
		sum += pow(x - avg, 2);
	}
	return sqrt(sum / v.size());
};

vector<float> convertShortToFloatVector(const std::vector<short>& shortVec) {
	vector<float> floatVec;
	floatVec.reserve(shortVec.size());

	for (const auto& element : shortVec) {
		floatVec.push_back(static_cast<float>(element));
	}

	return floatVec;
}


PreprocessedData normalizeTorch(torch::Tensor& t_data, NormMethod norm_method) {

	PreprocessedData preprocessed_data = PreprocessedData();

	torch::Tensor t_norm_data;

	if (norm_method == NormMethod::MINMAX) {
		//do min max normalization in accordance to python 
		INFO("Using MinMax normalization: ");
		torch::Tensor min = t_data.min();
		torch::Tensor max = t_data.max();
		t_norm_data = (t_data - min) / (max - min);
		preprocessed_data.t_data = t_norm_data;
		preprocessed_data.norm_params.min = min.item<float>();
		preprocessed_data.norm_params.max = max.item<float>();

		return preprocessed_data;
	}
	else if (norm_method == NormMethod::ZSCORE) {
		INFO("Using ZSCORE normalization: ");
		torch::Tensor mean = t_data.mean();
		torch::Tensor std = t_data.std();

		PRINT("mean val: " << mean);
		PRINT("std val: " << std);
		t_norm_data = (t_data - mean) / std;
		preprocessed_data.t_data = t_norm_data;
		preprocessed_data.norm_params.avg = mean.item<float>();
		preprocessed_data.norm_params.std = std.item<float>();

		return preprocessed_data;
	}
	else if (norm_method == NormMethod::PERCENTILE) {
		INFO("Using PERCENTILE normalization: ");
		float eps = 1e-10;

		auto t_mean = t_data.mean();
		auto t_std = t_data.std();
		//auto tensor_neg2std = torch::where(tensor < t_mean - (2 * t_std), t_mean - (2 * t_std), tensor);

		auto percentile0 = calculatePercentile(t_data, 0);
		auto percentile99 = calculatePercentile(t_data, 99);

		auto normalized_a = (t_data - percentile0) / ((percentile99 - percentile0) + eps);

		//PRINT("normalized_a" << normalized_a);
		preprocessed_data.t_data = normalized_a;
		preprocessed_data.norm_params.percentile0 = percentile0;
		preprocessed_data.norm_params.percentile99 = percentile99;

		return preprocessed_data;
	}

	else if (norm_method == NormMethod::NONE) {
		// do nothing 
		INFO("Using NONE normalization: ");
		preprocessed_data.t_data = t_data;
		return preprocessed_data;
	}

	else {
		ERR("Error: Invalid normalization method. ");
		return preprocessed_data;
	}

}
float calculatePercentile(torch::Tensor& tensor, float percentile) {
	auto sorted_data = torch::sort(tensor.flatten());
	Tensor sorted_values = std::get<0>(sorted_data);
	Tensor sorted_indices = std::get<1>(sorted_data);
	int index = (percentile / 100.0) * sorted_values.size(0);
	return sorted_values[index].item<float>();
}
Tensor resizeKeepRatioXray(Tensor& t_img, const int target_size) {
	std::vector<int64_t> old_size = t_img.sizes().vec();

	float ratio = static_cast<float>(target_size) / std::max(old_size[1], old_size[2]);
	std::vector<int64_t> new_size = { static_cast<int64_t> (old_size[1] * ratio), static_cast<int64_t>(old_size[2] * ratio) };
	t_img = nn::functional::interpolate(t_img.unsqueeze(0), nn::functional::InterpolateFuncOptions().size(new_size).mode(kNearest)).squeeze(0);

	return t_img;
}
Tensor resizeXray(Tensor& t_img, const int target_size) {
	std::vector<int64_t> old_size = t_img.sizes().vec();
	std::vector<int64_t> new_size = { target_size, target_size };
	t_img = nn::functional::interpolate(t_img.unsqueeze(0), nn::functional::InterpolateFuncOptions().size(new_size).mode(kNearest)).squeeze(0);

	return t_img;
}
Tensor padImageXray(torch::Tensor& t_img, const int target_size, const int pad_value) {

	int64_t channels = t_img.size(0);
	int64_t old_height = t_img.size(1);
	int64_t old_width = t_img.size(2);

	int64_t pad_size_h = target_size - old_height;
	int64_t pad_size_w = target_size - old_width;

	int64_t pad_top = pad_size_h / 2;
	int64_t pad_bottom = pad_size_h - pad_top;
	int64_t pad_left = pad_size_w / 2;
	int64_t pad_right = pad_size_w - pad_left;
	vector<int64_t> pad = { pad_left, pad_right, pad_top, pad_bottom };

	return F::pad(t_img, F::PadFuncOptions(pad).value(pad_value));
}