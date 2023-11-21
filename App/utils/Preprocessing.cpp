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
//vector<float_t> normalize1D(vector<float_t>& v_data, NormParams norm_params, NormMethod norm_method) {
//
//	std::vector<float_t> v_norm_data;
//	if (norm_method == NormMethod::NONE) {
//		// do nothing 
//		;
//		return v_data;
//	}
//
//	if (norm_method == NormMethod::MINMAX) {
//		//do min max normalization in accordance to python 
//		INFO("Using MinMax normalization: ");
//		double_t rs = (norm_params.max - norm_params.min);		//rescale slop
//		double_t ri = (norm_params.min);						//rescale intercept
//		for (float_t x : v_data) {
//			auto norm_val = (x - ri) / rs;
//			v_norm_data.push_back(norm_val);
//		}
//	}
//	if (norm_method == NormMethod::ZSCORE) {
//		INFO("Using Z-SCORE normalization: ");
//		double_t rs = norm_params.std;						//rescale slop
//		double_t ri = norm_params.avg;						//rescale intercept
//		for (float_t x : v_data) {
//			auto val = (x - ri) / rs;
//			v_norm_data.push_back(val);
//		};
//	}
//
//	return v_norm_data;
//
//}
////Result readRawFile1D_v2(std::string& raw_file_path, int d, int w, int h) {
////	std::ifstream input_file(raw_file_path, std::ios::in | std::ios::binary);
////	
////	std::vector<short> v_pixel_val;
////	float min = INFINITY;
////	float max = -INFINITY;
////	if (!input_file) {
////		ERR("Error: Failed to open file." << raw_file_path);
////		ERR("Returning an empty vector. ");
////		return { v_pixel_val, 0, 0}; //return empty vector nothing
////	};
////
////	if (input_file.is_open()) {
////		int16_t data; // size of the dicom data is 16 bit (2bytes)
////
////		int count = 0;
////		int img_resolution = d * h * w;
////		while (input_file.read(/*buffer*/(char*)&data, /*buffer size*/sizeof(data))) {
////			
////			v_pixel_val.push_back((short)(data));
////			if (data < min) {
////				min = data;
////			}
////			if (data > max) {
////				max = data;
////			}
////			++count;
////			if (count > img_resolution) {
////				throw std::invalid_argument(std::string("Wrong resolution value. Image size is larger than ") +
////					std::to_string(h) + " x " + std::to_string(w) + "x" + std::to_string(d));
////			}
////		};
////		
////		input_file.close();
////	}
////	printf("Reading file successful.\n");
////	return { v_pixel_val, min, max };
////};
//// ----------------------------- TENSOR OPERATIONS ------------------------------------
//// overloaded for Tensor normalization
Tensor normalizeTorch(torch::Tensor& t_data, NormMethod norm_method) {


	torch::Tensor t_norm_data;

	if (norm_method == NormMethod::MINMAX) {
		//do min max normalization in accordance to python 
		INFO("Using MinMax normalization: ");
		torch::Tensor min = t_data.min();
		torch::Tensor max = t_data.max();
		t_norm_data = (t_data - min) / (max - min);
		return t_norm_data;
	}
	else if (norm_method == NormMethod::ZSCORE) {
		INFO("Using ZSCORE normalization: ");
		torch::Tensor mean = t_data.mean();
		torch::Tensor std = t_data.std();

		PRINT("mean val: " << mean);
		PRINT("std val: " << std);
		t_norm_data = (t_data - mean) / std;
		return t_norm_data;
	}
	else if (norm_method == NormMethod::PERCENTILE) {
		float eps = 1e-10;

		auto t_mean = t_data.mean();
		auto t_std = t_data.std();
		//auto tensor_neg2std = torch::where(tensor < t_mean - (2 * t_std), t_mean - (2 * t_std), tensor);

		auto percentile0 = calculatePercentile(t_data, 0);
		auto percentile99 = calculatePercentile(t_data, 99);

		auto normalized_a = (t_data - percentile0) / ((percentile99 - percentile0) + eps);

		//PRINT("normalized_a" << normalized_a);

		return normalized_a;
	}

	else if (norm_method == NormMethod::NONE) {
		// do nothing 
		INFO("Using NONE normalization: ");
		return t_data;
	}

	else {
		ERR("Error: Invalid normalization method. ");
		return t_data;
	}

}
float calculatePercentile(torch::Tensor& tensor, float percentile) {
	auto sorted_data = torch::sort(tensor.flatten());
	Tensor sorted_values = std::get<0>(sorted_data);
	Tensor sorted_indices = std::get<1>(sorted_data);
	int index = (percentile / 100.0) * sorted_values.size(0);
	return sorted_values[index].item<float>();
}
//PatchData extractPatches(Tensor& input,
//	Tensor(*normFun)(Tensor&, NormMethod),
//	NormMethod norm_method,
//	const int64_t patch_size,
//	const int64_t stride
//)
//{
//	/*
//	The input tensor must be of size [x_size, y_size, z_size]
//	*/
//	if (input.sizes().size() != 3) {
//		LOG(FATAL) << "The input tensor must be of size [x_size, y_size, z_size]";
//	}
//	//TODO: Double check the input tensor size (h,w) switch
//	input = input.permute({ 2, 1, 0 });
//
//	int64_t constant = -1024;
//	int64_t x_size = input.sizes()[0];
//	int64_t y_size = input.sizes()[1];
//	int64_t z_size = input.sizes()[2];
//
//	int64_t x_pad = x_size / patch_size + 1;
//	int64_t y_pad = y_size / patch_size + 1;
//	int64_t z_pad = z_size / patch_size + 1;
//
//	x_pad = x_pad * patch_size - x_size;
//	y_pad = y_pad * patch_size - y_size;
//	z_pad = z_pad * patch_size - z_size;
//	std::vector<int64_t> pad_size = { x_pad, y_pad, z_pad };
//	//torch::IntArrayRef pad = { z_pad, 0, y_pad, 0, x_pad, 0 };
//	std::vector <int64_t> pad = { z_pad, 0, y_pad, 0, x_pad, 0 };
//	PRINT( "input dims: " << input.sizes());
//	//Tensor padded_input = torch::constant_pad_nd( input, pad, constant );
//	Tensor padded_input = F::pad(input, F::PadFuncOptions(pad).value(constant));
//	PRINT("padded input dims: " << padded_input.sizes());
//	
//	
//
//	// Normalize input value
//	padded_input = normFun(padded_input, norm_method);
//	Tensor patches = padded_input.unfold(0, patch_size, stride).unfold(1, patch_size, stride).unfold(2, patch_size, stride);
//	std::vector<int64_t> unfold_shape(patches.sizes().begin(), patches.sizes().end());
//	patches = patches.contiguous().view({ -1, patch_size, patch_size, patch_size });
//	return { patches, pad_size, unfold_shape };
//}
Tensor reconPatches(Tensor& patches, const std::vector<int64_t>& unfold_shape) {
	auto patches_orig = patches.view({ unfold_shape });
	auto output_c = unfold_shape[0] * unfold_shape[3];
	auto output_h = unfold_shape[1] * unfold_shape[4];
	auto output_w = unfold_shape[2] * unfold_shape[5];
	patches_orig = patches_orig.permute({ 0,3,1,4,2,5 }).contiguous();
	patches_orig = patches_orig.view({output_c, output_h, output_w});
	return patches_orig;
}
Tensor resizeKeepRatioXray(Tensor& t_img, const int target_size) {
	std::vector<int64_t> old_size = t_img.sizes().vec();

	float ratio = static_cast<float>(target_size) / std::max(old_size[1], old_size[2]);
	std::vector<int64_t> new_size = { static_cast<int64_t> (old_size[1] * ratio), static_cast<int64_t>(old_size[2] * ratio) };
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