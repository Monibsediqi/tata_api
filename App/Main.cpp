/*
Author: Monib Sediqi @ Medical IP Inc.
Email: monib.sediqi@medicalip.com | kh.monib@gmail.com
Date: 2023-02-01
All rights reserved.
*/

// system includes
#include <chrono>
#include <tuple>

// local includes
#include "utilities/Preprocessing.h"
#include "utilities/Postprocessing.h"
#include "utilities/Security.h"
#include "stdx.h"

// third party includes
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>

#include <openssl/evp.h>
#include <openssl/rand.h>
#undef max;
#undef min;


#define PRINT(x) std::cout << x << std::endl;
#define INFO(x) std::cout << "INFO: " << x << std::endl;
#define ERR(x) std::cerr << "ERROR: " << x << std::endl;


using namespace torch;
using namespace std;
namespace F = torch::nn::functional;

class ConvBlock1Impl : public torch::nn::Module {
public:
	ConvBlock1Impl(int in_channels, int out_channels, float drop_prob) {
		_conv1 = register_module("_conv1", torch::nn::Conv3d(torch::nn::Conv3dOptions(out_channels, out_channels, 3).padding(1).bias(false)));
		_norm1 = register_module("_norm1", torch::nn::InstanceNorm3d(nn::InstanceNorm3dOptions(out_channels)));
		_leaky_relu1 = register_module("_leaky_relu1", torch::nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.1)));
		_dropout1 = register_module("_dropout1", torch::nn::Dropout3d(nn::Dropout3dOptions(drop_prob)));
	};
	torch::Tensor forward(torch::Tensor x) {
		x = _conv1->forward(x);
		x = _norm1->forward(x);
		x = _leaky_relu1->forward(x);
		x = _dropout1->forward(x);
		return x;
	}

private:
	nn::Conv3d _conv1{ nullptr };
	nn::InstanceNorm3d _norm1{ nullptr };
	nn::LeakyReLU _leaky_relu1{ nullptr };
	nn::Dropout3d _dropout1{ nullptr };

}; TORCH_MODULE(ConvBlock1);

class ConvBlock2Impl : public torch::nn::Module {
public:
	ConvBlock2Impl(int in_channels, int out_channels, float drop_prob) {
		_convblock = register_module("_convblock", ConvBlock1(in_channels, out_channels, drop_prob));
		_conv2_1 = register_module("_conv2_1", torch::nn::Conv3d(torch::nn::Conv3dOptions(in_channels, out_channels, 3).padding(1).bias(false)));
		_norm2_1 = register_module("_norm2_1", torch::nn::InstanceNorm3d(nn::InstanceNorm3dOptions(out_channels)));
		_leaky_relu2_1 = register_module("_leaky_relu2_1", torch::nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.1)));
		_dropout2_1 = register_module("_dropout2_1", torch::nn::Dropout3d(nn::Dropout3dOptions(drop_prob)));

	};
	torch::Tensor forward(torch::Tensor x) {
		x = _convblock->forward(x);
		x = _conv2_1->forward(x);
		x = _norm2_1->forward(x);
		x = _leaky_relu2_1->forward(x);
		x = _dropout2_1->forward(x);
		return x;
	}

private:
		ConvBlock1 _convblock{ nullptr };
		nn::Conv3d _conv2_1{ nullptr };
		nn::InstanceNorm3d _norm2_1{ nullptr };
		nn::LeakyReLU _leaky_relu2_1{ nullptr };
		nn::Dropout3d _dropout2_1{ nullptr };
}; TORCH_MODULE(ConvBlock2);

INITIALIZE_EASYLOGGINGPP


void initializeLoggerV2(std::string conf_path) {
	// Load configuration from file
	el::Configurations conf(conf_path);
	// Reconfigure single logger
	el::Loggers::reconfigureLogger("default", conf);
	// Actually reconfigure all loggers instead
	el::Loggers::reconfigureAllLoggers(conf);
	// Now all the loggers will use configuration from file
}

// ---------------------- TISEPX ----------------------
// Breif version of Dicom data in tensor format
struct DICOM_T {
	Tensor data_TS; // Tensor Data
	Tensor pixelspacing;
	Tensor spatialresolution;
	Tensor age;
	Tensor photometricInterpretation;
	Tensor pixelRepresentation;
	Tensor bitsAllocated;
	Tensor bitsStored;
	Tensor highBit;
	Tensor rescaleIntercept;
	Tensor rescaleSlope;
	Tensor windowCenter;
	Tensor windowWidth;
	Tensor imagePositionPatient;
	Tensor imageOrientationPatient;
	Tensor sliceLocation;
	Tensor sliceThickness;
	Tensor spacingBetweenSlices;
	Tensor rows;
	Tensor columns;
};



//cv::Mat histogram_normalization(cv::Mat arr) {
//	try {
//		arr.convertTo(arr, CV_32F);
//		cv::Mat a_norm = ((arr - cv::min(arr)) / (cv::max(arr) - cv::min(arr)) * 255).toMat();
//		if (a_norm.channels() == 4) {
//			a_norm = a_norm(cv::Rect(0, 0, 1, 1));
//		}
//		else if (a_norm.channels() == 3) {
//			a_norm = a_norm(cv::Rect(0, 0, 1));
//		}
//		cv::Mat a_norm_new;
//		cv::cvtColor(a_norm, a_norm_new, cv::COLOR_GRAY2BGR);
//
//		std::vector<cv::Mat> channels(3);
//		cv::split(a_norm_new, channels);
//		cv::Mat a_norm_tile;
//		cv::merge(channels, a_norm_tile);
//
//		int histSize = 256;
//		float range[] = { 0, 256 };
//		const float* histRange = { range };
//		cv::Mat hist;
//		cv::calcHist(&a_norm_tile, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
//
//		cv::Mat cdf;
//		cv::Mat cdf_m;
//		cv::reduce(hist, cdf, 0, cv::REDUCE_SUM);
//		cv::minMaxLoc(cdf, nullptr, nullptr, nullptr, &cdf_m);
//		cdf = (cdf_m - cv::min(cdf_m)) * 255 / (cv::max(cdf_m) - cv::min(cdf_m));
//		cdf.convertTo(cdf, CV_8U);
//
//		cv::Mat arr_histnorm;
//		cv::LUT(a_norm_tile, cdf, arr_histnorm);
//		cv::max()
//		cv::Mat arr_denorm = (arr_histnorm / 255) * (cv::max(arr) - cv::min(arr)) + cv::min(arr);
//
//		cv::Mat result;
//		cv::extractChannel(arr_denorm, result, 0);
//
//		return result;
//	}
//	catch (...) {
//		return arr;
//	}
//}

//torch::Tensor histogramNormalization(torch::Tensor arr) {
//	try {
//		arr = arr.to(torch::kFloat);
//		auto arr_min = arr.min().item<float>();
//		auto arr_max = arr.max().item<float>();
//
//		auto a_norm = ((arr - arr_min) / (arr_max - arr_min) * 255).to(torch::kInt);
//
//		if (a_norm.dim() == 4) {
//			a_norm = a_norm.squeeze(2).squeeze(2);
//		}
//		else if (a_norm.dim() == 3) {
//			a_norm = a_norm.squeeze(2);
//		}
//		PRINT("a_norm size: " << a_norm.sizes())
//
//		a_norm = a_norm.expand({ a_norm.size(0), a_norm.size(1), 3 });
//
//		auto hist = torch::histc(a_norm.flatten(), 256, 0, 256);
//		auto cdf = hist.cumsum(0);
//
//		auto cdf_m = cdf.masked_fill_(cdf == 0, 1);
//		cdf_m = ((cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())).to(torch::kUInt8);
//
//		auto cdf_filled = cdf_m.fill_(0);
//		//auto cdf_filled = cdf_m.filled(0);
//		auto arr_histnorm = torch::index_select(cdf_filled, 0, a_norm.flatten().to(torch::kLong)).reshape_as(a_norm);
//
//		auto arr_denorm = (arr_histnorm / 255) * (arr_max - arr_min) + arr_min;
//
//		return arr_denorm.select(2, 0);
//	}
//	catch (...) {
//		return arr;
//	}
//}

class ImageResizer {
public:

	ImageResizer(int target_size) : target_size(target_size) {}
	cv::Mat resize_keep_ratio(const cv::Mat& img) {
		cv::Size old_size = img.size();

		float ratio = static_cast<float>(target_size) / std::max(old_size.width, old_size.height);
		cv::Size new_size(static_cast<int> (old_size.width * ratio), static_cast<int>(old_size.height * ratio));
		cv::Mat resized_image;
		cv::resize(img, resized_image, new_size, 0, 0, cv::INTER_NEAREST);
		return resized_image;
	}
	
private:
	int target_size;
};

cv::Mat pad_image(cv::Mat& img, int load_size /* width & height*/, int pad_value = -1024) {

	cv::Size old_size = img.size();
	cv::Size target_size = cv::Size(load_size, load_size);

	float pad_size_w = (target_size.width - old_size.width) / 2;
	float pad_size_h = (target_size.height - old_size.height) / 2;

	int wl, wr, ht, hb;
	if (fmod(pad_size_w, 2.0) == 0) {
		wl = wr = static_cast<int>(pad_size_w);
	}
	else {
		wl = ceil(pad_size_w);
		wr = floor(pad_size_w);
	}
	if (fmod(pad_size_h, 2.0) == 0) {
		ht = hb = static_cast<int>(pad_size_h);
	}
	else {
		ht = ceil(pad_size_h);
		hb = floor(pad_size_h);
	}
	cv::Scalar padColor(pad_value);
	cv::Mat padded_img;
	cv::copyMakeBorder(img, padded_img, ht, hb, wl, wr, cv::BORDER_CONSTANT, padColor);
	return padded_img;

};

Tensor mat2Tensor(const cv::Mat& img) {
	// convert a 2D CV image into Tensor 
	// input: 2D CV image
	// output: 2D Tensor
	cv::Mat img_float;
	img.convertTo(img_float, CV_32F); // convert to float32

	Tensor tensor_img = torch::from_blob(img_float.data, { img_float.rows, img_float.cols, 1 /*img_float.channels*/ }, torch::kFloat);
	tensor_img = tensor_img.permute({ 2, 0, 1 }); // convert to CxHxW

	return tensor_img.clone();
}

cv::Mat tensor2mat(Tensor tensor) {
	// convert a 2D Tensor into CV image
	// input: 2D Tensor
	// output: 2D CV image

	tensor = tensor.permute({ 1, 2, 0 }); // convert to HxWxC
	int cvType;
	if (tensor.dtype() == torch::kFloat32) {
		cvType = CV_32F;
	}
	else if (tensor.dtype() == torch::kUInt8) {
		cvType = CV_8U;
	}
	else if (tensor.dtype() == torch::kInt32) {
		cvType = CV_32S;
	}
	else {
		throw std::runtime_error("tensor2mat(): tensor type is not supported.");
	}

	int height = tensor.size(0);
	int width = tensor.size(1);
	tensor = tensor.mul(255).clamp(0, 255).to(torch::kUInt8);

	cv::Mat img(height, width, CV_8UC1, tensor.data_ptr());

	//// Convert the tensor to 8-bit unsigned integer (CV_8U) for proper visualization
	//T = tensor.mul(255).clamp(0, 255).to(torch::kUInt8);

	//// Get the image dimensions
	//int height = T.size(0);
	//int width = T.size(1);

	//// Create a cv::Mat object with the same dimensions and data type as the tensor
	//cv::Mat image(height, width, CV_8UC1, T.data_ptr());

	// Display the image
	//cv::imshow("X-ray Image", image);
	cv::imwrite("img test.jpg", img);

	return img;
}

float calculatePercentile(torch::Tensor& tensor, float percentile) {
	auto sorted_data = torch::sort(tensor.flatten());
	Tensor sorted_values = std::get<0>(sorted_data);
	Tensor sorted_indices = std::get<1>(sorted_data);
	int index = (percentile / 100.0) * sorted_values.size(0);
	return sorted_values[index].item<float>();
}

Tensor percentile_norm(Tensor& tensor) {

	float eps = 1e-10;

	auto t_mean = tensor.mean();
	auto t_std = tensor.std();
	//auto tensor_neg2std = torch::where(tensor < t_mean - (2 * t_std), t_mean - (2 * t_std), tensor);
	
	auto percentile0 = calculatePercentile(tensor, 0);
	auto percentile99 = calculatePercentile(tensor, 99);
	
	auto normalized_a = (tensor - percentile0) / ((percentile99 - percentile0) + eps);

	//PRINT("normalized_a" << normalized_a);

	return normalized_a;
}


Tensor predict_xray2lung(Tensor& tensor, jit::script::Module& model) {
	
	auto t_data = tensor.unsqueeze(0);
	jit::IValue input = t_data;
	


	//if (IS_HALF_PRECISION) {
	//	input = t_data.unsqueeze(0).to(device).to(kHalf);
	//}
	//else {
	//	input = t_data.unsqueeze(0).to(device).to(kFloat); // The defual kFloat is 32 bits
	//}
	torch::NoGradGuard no_grad;
	auto output = model.forward({input}); // ouput= {1, 1, 2048, 2048}
	auto t_output = output.toTensor().to(kCPU); // ouput= {1, 1, 2048, 2048}
	PRINT("t_output shape: " << t_output.sizes());
	return t_output;
	
}

Tensor denormalize(Tensor& tensor, int b_min_value = -1100, int b_max_value= -500) {

	return tensor * (b_max_value - b_min_value) + b_min_value;
}

Tensor resize_keep_ratio(Tensor& img, int target_size, int pad_value = -1024) {
std::vector<int64_t> old_size = img.sizes().vec(); 
PRINT("old size vector: " << old_size)

float ratio = static_cast<float>(target_size) / std::max(old_size[1], old_size[2]);
std::vector<int64_t> new_size = {static_cast<int64_t> (old_size[1] * ratio), static_cast<int64_t>(old_size[2] * ratio)};

Tensor im;

im = nn::functional::interpolate(img.unsqueeze(0), nn::functional::InterpolateFuncOptions().size(new_size).mode(kNearest)).squeeze(0);

return im;
}

Tensor pad_image_torch(torch::Tensor img, int load_size, int pad_value = -1024) {
	int64_t channels = img.size(0);
	int64_t old_height = img.size(1);
	int64_t old_width = img.size(2);

	int64_t pad_size_h = load_size - old_height;
	int64_t pad_size_w = load_size - old_width;

	int64_t pad_top = pad_size_h / 2;
	int64_t pad_bottom = pad_size_h - pad_top;
	int64_t pad_left = pad_size_w / 2;
	int64_t pad_right = pad_size_w - pad_left;
	vector<int64_t> pad = { pad_left, pad_right, pad_top, pad_bottom};
	PRINT("pad: " << pad)

	Tensor padded_img = F::pad(img, F::PadFuncOptions(pad).value(pad_value));

		/*torch::full({ channels, load_size, load_size }, pad_value);
	padded_img.index_put_({ indexing::Slice(pad_top, pad_top + old_height),
						   indexing::Slice(pad_left, pad_left + old_width) },img);*/

	return padded_img;
}

Tensor get_mask_area(Tensor& tensor, float pixel_size_resize_w, float pixel_size_resize_h) {
	auto area = torch::sum(tensor);
	return area * pixel_size_resize_w * pixel_size_resize_h / 100;
};


int main(int argc, const char* argv[]) {

	// CONSTANTS 
	bool IS_HALF_PRECISION = false;

	// Raw CT dimension (for test purpose only)
	int CT_D = 256;
	int CT_W = 256;
	int CT_H = 256;

	// Raw X-Ray dimension (for test purpose only)
	int XRAY_D = 1;
	int XRAY_W = 2652;
	int XRAY_H = 2450;
	
	 
	// Images Path
	// -------------------------------- XRAY RAW FILES PATH -----------------------------------------
	// 
	string LUNG_VOL_RAW = "F:\\2023\\AI\\sample_data\\tisepx_a\\lung_vol_raw\\JB0006_CXR_0base_201229.raw";


	// ------------------------------- SEGMENTATION RAW FILES PATH ---------------------------
	// sample image 
	string  SAMPLE_CT_RAW_PATH = "F:\\2023\\AI\\sample_data\\image_a\\case24_256.raw";

	// Models Path
	// ------------------------------- TISEPX MIPX MODELS -------------------------------------------
	string XRAY2LUNG_CPU = "F:\\2023\\AI\\App\\scripted_models\\new_xray2lung.mipx";
	string LUNG_REG = "F:\\2023\\AI\\App\\scripted_models\\lungregression_cpu.mipx";
	string TISEP_TB_CUDA = "F:\\2023\\AI\\App\\scripted_models\\tiseptb2_cuda.mipx";


	// ------------------------------- MEDIP PRO SEGMENTATION MODELS --------------------------------
	string SWIN_UNETR_DEEPCATCH_CPU = "F:\\2023\\AI\\App\\scripted_models\\traced_model_cpu.pt";
	string SWIN_UNETR_DEEPCATCH_CUDA = "F:\\2023\\AI\\App\\scripted_models\\traced_model_cuda.pt";
	string SWIN_UNETR_DEEPCATCH_CUDA_HALF_PRECISION = "F:\\2023\\AI\\App\\scripted_models\\half_traced_model_cuda.pt";



	// ------------------------------- TESTING THE CV FUNCTIONS -------------------------------------
	/*string img_path = "F:\\2023\\AI\\sample_data\\jpg\\AN_ID_20210526104509_1.jpg";
	cv::Mat img = cv::imread(img_path);
	ImageResizer resizer (2048);
	cv::Mat resized_img = resizer.resize_keep_ratio(img);

	PRINT("Original Image Size: " << img.size());
	PRINT("Resized Image Size: " << resized_img.size());

	cv::Mat padded_img = pad_image(resized_img, 2048, 0);
	PRINT("Padded Image Size: " << padded_img.size());*/



#ifdef _EXPERIMENT_

	std::cout << "We are in _EXPERIMENT_ mode" << endl;
	//int img_h = 256;
	//int img_w = 256;
	//int img_d = 184;


	//Tensor a = torch::rand({ img_h, img_w, 184}); // creating patches from multipels of 96


	//int64_t patch_size = 96;
	//int64_t stride = 96;

	//PatchData patch_data = extractPatches(a, &normalizeTorch, NormMethod::ZSCORE, patch_size, stride);
	//std::vector<Tensor> patches = patch_data.patches;
	//std::vector<int64_t> pad_size = patch_data.pad_size;
	//Tensor unfold_shape = patch_data.out_shape;

	//// cout
	//std::cout <<"patches: " << patches.size() << endl;
	//std::cout <<"pad size: "  <<pad_size[0] << " " <<pad_size[1] << " " <<pad_size[2] << endl;
	//std::cout <<"out shape: "  << unfold_shape.sizes() << endl;

	//INFO("Number of patches: " << patches.size());


	////std::vector<int64_t> v_temp = { 6,6,6, 96,96,96 };
	//std::vector<int64_t> v_unfold_shape(unfold_shape.sizes().begin(), unfold_shape.sizes().end());
	////Tensor combined_patches = torch::zeros({ 6 * 6 * 6, 96,96,96 });

	//Tensor combined_patches = torch::zeros({(long long) patches.size(), patch_size, patch_size, patch_size });
	//std::cout << "combined patches shape: " << combined_patches.sizes() << endl;
	//for (int i = 0; i < combined_patches.size(0); i++) {
	//	//std::cout << "i: " << i<<endl;
	//	combined_patches[i] = patches[i];
	//};

	
	// reconstruct the image from patches
	//Tensor recon_img = reconPatches(combined_patches, v_unfold_shape);
	//std::cout << "recon img v2 shape: " << recon_img.sizes() << endl;

	//// remove padding from the front of x,y,z dims.
	//Tensor seg_mask = recon_img.slice(0, pad_size[0] , recon_img.sizes()[0]).slice(1, pad_size[1],  recon_img.sizes()[1]).slice(2, pad_size[2], recon_img.sizes()[2]);
	//std::cout << "reco image after removing the padding: " << seg_mask.sizes() << endl;

	////Convert tensor into a 3D vector
	//std::vector<std::vector<std::vector<int16_t>>> vec_output = convertTensorTo3DVectorInt16(seg_mask);

	//Save the converted tensor into a raw file
	//writeRawData3D(vec_output, "..\\sample_data\\image_b\\[fake]case57_256.raw");
	//INFO("Done!");

	//return 0;
	//int img_h = 256;
	//int img_w = 256;
	//int img_d = 184;
	//int64_t patch_size = 96;
	//int64_t stride = 96;

	//Tensor a = torch::rand({ img_h, img_w, 184}); // creating patches from multipels of 96


	//PatchData patch_data = extractPatches(a, &normalizeTorch, NormMethod::ZSCORE, patch_size, stride);
	//std::vector<Tensor> patches = patch_data.patches;
	//std::vector<int64_t> pad_size = patch_data.pad_size;
	//Tensor unfold_shape = patch_data.out_shape;


	//std::cout <<"patches: " << patches.size() << endl;
	//std::cout <<"pad size: "  <<pad_size[0] << " " <<pad_size[1] << " " <<pad_size[2] << endl;
	//std::cout <<"out shape: "  << unfold_shape.sizes() << endl;

	//INFO("Number of patches: " << patches.size());

	//std::vector<int64_t> v_unfold_shape(unfold_shape.sizes().begin(), unfold_shape.sizes().end());


	//Tensor combined_patches = torch::zeros({(long long) patches.size(), patch_size, patch_size, patch_size });
	//std::cout << "combined patches shape: " << combined_patches.sizes() << endl;
	//for (int i = 0; i < combined_patches.size(0); i++) {
	//	//std::cout << "i: " << i<<endl;
	//	combined_patches[i] = patches[i];
	//};

	//
	//// reconstruct the image from patches
	//Tensor recon_img = reconPatches(combined_patches, v_unfold_shape);
	//std::cout << "recon img v2 shape: " << recon_img.sizes() << endl;

	//// remove padding from the front of x,y,z dims.
	//Tensor seg_mask = recon_img.slice(0, pad_size[0] , recon_img.sizes()[0]).slice(1, pad_size[1],  recon_img.sizes()[1]).slice(2, pad_size[2], recon_img.sizes()[2]);
	//std::cout << "reco image after removing the padding: " << seg_mask.sizes() << endl;

	//////Convert tensor into a 3D vector
	//std::vector<std::vector<std::vector<int16_t>>> vec_output = convertTensorTo3DVectorInt16(seg_mask);

	////// Save the converted tensor into a raw file
	//writeRawData3D(vec_output, "..\\sample_data\\image_b\\[fake]case57_256.raw");
	//INFO("Done!");

	//return 0;

#else 

	INFO("We are  not in _EXPERIMENT_ mode\n");
 
	Device device (kCUDA, 0); //torch::kCUDA if you have a GPU
	torch::jit::script::Module module;

	// --------------------------------------- MODEL ENCRYPTION/DECRYPTION UTILITY FUNCTION ---------------------------------
	/*string secret_key = "This_is_secret";
	string encrypted_model = "encrypted_model.bin";
	string decrypted_model = "decrypted_model.pt";


	encrypt_file(half_precision_scripted_swinUNetR_path, encrypted_model, secret_key);
	decrypt_file(encrypted_model, decrypted_model, secret_key);*/


	//check if gpu is available
	if (torch::cuda::is_available()) {
		INFO("CUDA is available! Using GPU. FIXME ... ");
		device = kCUDA;
	}
	else {
		INFO("CUDA is not available. Using CPU.");
		device = kCPU;
	}
	
	if (XRAY2LUNG_CPU.empty()) {
		ERR("No such file " << XRAY2LUNG_CPU);
		return -1;
	}
	
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		if (IS_HALF_PRECISION) {
			INFO("Using half precision!")
			module = torch::jit::load(XRAY2LUNG_CPU, device);
			module.eval();
		}
		else {
			INFO("Using full precision!")
			module = torch::jit::load(XRAY2LUNG_CPU, device);
			module.eval();
		}
	}
	catch (exception e) {
		ERR(e.what());
		return -1;
	}
	INFO("Loading model successful.\n");

	vector<short> vec_img_A = readRawFile1D(LUNG_VOL_RAW, XRAY_D, XRAY_W, XRAY_H); // read image A

	vector<float_t> vec_img_A_float(vec_img_A.begin(), vec_img_A.end());

	//convert to torch tensor 
	Tensor t_data = torch::from_blob(vec_img_A_float.data(), IntArrayRef{XRAY_D, XRAY_H, XRAY_W }, kFloat32);
	//PRINT("t_data size" << t_data.sizes());
	
	//t_data = histogramNormalization(t_data);

	Tensor t_resized = resize_keep_ratio(t_data, 2048);
	//PRINT("resized data " << t_resized.sizes())
	//vector<vector<vector<int16_t>>> padded_cxr_v2 = convertTensorTo3DVectorInt16(t_resized);
	////Save the converted tensor into a raw file
	//writeRawData3D_v2(padded_cxr_v2, "F:\\2023\\AI\\sample_data\\tisepx_b\\t_resized.raw");

	Tensor padded_img = pad_image_torch(t_resized, 2048, 0);

	//PRINT("padded_img data " << padded_img.sizes())
	//vector<vector<vector<int16_t>>> padded_img_v = convertTensorTo3DVectorInt16(padded_img);
	////Save the converted tensor into a raw file
	//writeRawData3D_v2(padded_img_v, "F:\\2023\\AI\\sample_data\\tisepx_b\\t_resized.raw");

	/*cv::Mat cv_img = tensor2mat(t_data);
	ImageResizer resizer_cxr(2048);
	cv::Mat resized_cxr = resizer_cxr.resize_keep_ratio(cv_img);
	cv::Mat padded_cxr = pad_image(resized_cxr, 2048, 0);*/
	padded_img = percentile_norm(padded_img);
	float eps = 1e-10;
	padded_img = padded_img.to(device);
	
	//padded_img = (padded_img - padded_img.min().item<float>()) / ((padded_img.max().item<float>() - padded_img.min().item<float>()) + eps);

	//vector<vector<vector<int16_t>>> norm_v = convertTensorTo3DVectorInt16(padded_img);
	////Save the converted tensor into a raw file
	//writeRawData3D_v2(norm_v, "F:\\2023\\AI\\sample_data\\tisepx_b\\norm_v.raw");


	//padded_img = normalizeTorch(padded_img, )

 	PRINT("padded_img after norm min " << torch::min(padded_img));
	PRINT("padded_img after norm max " << torch::max(padded_img));

	//vector<vector<vector<int16_t>>> padded_img_v = convertTensorTo3DVectorInt16(padded_img);
	////Save the converted tensor into a raw file
	//writeRawData3D_v2(padded_img_v, "F:\\2023\\AI\\sample_data\\tisepx_b\\padded_img_normalized.raw");
	

	//torch::jit::IValue input;
	Tensor pred = predict_xray2lung(padded_img, module);
	pred = torch::squeeze(pred, 0);
	PRINT("pred_xray2lung shape: " << pred.sizes())
	Tensor denorm_pred = denormalize(pred);


    // apply threshold 
	float threshold = -1015.0;
	float replace_value = -1024;
	denorm_pred = torch::where(denorm_pred[0] < threshold, torch::full_like(denorm_pred, replace_value), denorm_pred);

	auto denrom_pred_mask = torch::where(denorm_pred[0] < threshold, torch::zeros_like(denorm_pred[0]), torch::ones_like(denorm_pred[0]));
	PRINT("denrom_pred_mask shape: " << denrom_pred_mask.sizes());
	// print min max
	PRINT("denorm_pred min " << torch::min(denrom_pred_mask));
	PRINT("denorm_pred max " << torch::max(denrom_pred_mask));

	auto pixel_size_resize_w = 0.18517382812499997;
	auto pixel_size_resize_h = 0.18517382812499997;

	auto area = get_mask_area(denrom_pred_mask, pixel_size_resize_w, pixel_size_resize_h);
	PRINT("area " << area);

	// -------------------- regression part --------------------------
	// load model
	torch::jit::script::Module module_reg;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		if (IS_HALF_PRECISION) {
			INFO("Using half precision!");
			module_reg = torch::jit::load(LUNG_REG, device);
			module_reg.eval();
		}
		else {
			INFO("Using full precision!");
			module_reg = torch::jit::load(LUNG_REG, device);
			module_reg.eval();
		}
	}
	catch (exception e) {
		ERR(e.what());
		return -1;
	}
	INFO("Loading lung regression model successful.\n");


	//PRINT("Module: "<< module_reg);
	// FIXME: Constant values are temporary. Fix them for different input sizes.
	/*auto parameters = module_reg.named_parameters();

	PRINT("paramers: " << parameters.size());*/

	auto target_size = 1187.5878547836917;
	auto input_img = resize_keep_ratio(denorm_pred, target_size);
	PRINT("input_img shape : " << input_img.sizes());
	// print min-max
	PRINT("input_img min " << torch::min(input_img));
	PRINT("input_img max " << torch::max(input_img));

	auto input_img_padded = pad_image_torch(input_img, 2048);
	PRINT("input_img_padded shape  :" << input_img_padded.sizes());
	// print min-max
	PRINT("input_img_padded min " << torch::min(input_img_padded));
	PRINT("input_img_padded max " << torch::max(input_img_padded));
	// normalization 

	auto minHU = -1025.0;
	auto maxHU = -775.0;

	input_img_padded = torch::clip(input_img_padded, minHU, maxHU);
	//print min max
	PRINT("after clip input_img_padded min " << torch::min(input_img_padded));
	PRINT("after clip input_img_padded max " << torch::max(input_img_padded));

	// input norm
	input_img_padded = (input_img_padded - minHU) / (maxHU - minHU);
	//print min max
	PRINT("after norm input_img_padded min " << torch::min(input_img_padded));
	PRINT("after norm input_img_padded max " << torch::max(input_img_padded));

	auto sex = 1.0;
	auto age = 0.76;
	auto lung_area = area.item<float>();

	float input_values[] = { lung_area, sex, age };
	int64_t num_elements = sizeof(input_values) / sizeof(float);

	auto input_feat = torch::from_blob(input_values, { 1, num_elements }, kFloat).clone().to(device);

	input_img_padded = input_img_padded.unsqueeze(0).to(device);
	//print tensor shape
	PRINT("input_feat shape: " << input_feat.sizes()); // expecting 1x3
	PRINT("padded_img shape: " << input_img_padded.sizes()); // // expecting [1,1,2048,2048]
	
	PRINT("input feat " << input_feat);
	PRINT("input_img_padded " <<input_img_padded[0][0][0][0]);
	

	torch::jit::IValue inp_feat = input_feat;
	torch::jit::IValue in_img = input_img_padded;

	torch::NoGradGuard nograd;
	auto reg_pred = module_reg.forward({in_img, inp_feat}).toTensor().to(kCPU);
	//print output shape
	//PRINT("reg_pred shape: " << reg_pred.sizes()); // expecting 1x1x2048x2048
	PRINT("reg_pred value: " << reg_pred);


	Tensor T = denorm_pred;
	PRINT("T sizes " << T.sizes())

	vector<vector<vector<int16_t>>> vec_output1 = convertTensorTo3DVectorInt16(T);
	//Save the converted tensor into a raw file
	writeRawData3D_v2(vec_output1, "F:\\2023\\AI\\sample_data\\tisepx_b\\Test_T.raw");


	// Get the minimum and maximum values in the tensor
	float min_value = T.min().item<float>();
	float max_value = T.max().item<float>();

	// Scale the tensor values to the range [0, 1]
	T = (T - min_value) / (max_value - min_value);

	// Convert the tensor to 8-bit unsigned integer (CV_8U) for proper visualization
	T = T.mul(255).clamp(0, 255).to(torch::kUInt8);

	// Get the image dimensions
	int height = T.size(0);
	int width = T.size(1);

	// Create a cv::Mat object with the same dimensions and data type as the tensor
	cv::Mat image(height, width, CV_8UC1, T.data_ptr());

	// Display the image
	//cv::imshow("X-ray Image", image);
	cv::imwrite("Lung-Segmentation.jpg", image);
	//cv::waitKey(0);
	//cv::destroyAllWindows();
	
	/*PRINT("denorm_pred_seq shape:" << denorm_pred_seq.sizes());

	float eps = 1e-10;
	auto tensor_img = (denorm_pred_seq - denorm_pred_seq.min()) / ((denorm_pred_seq.max() - denorm_pred_seq.min()) + eps) * 255;
	denorm_pred_seq = denorm_pred_seq.mul(255).clamp(0, 255)
	cv::Mat final_img = tensor2mat(tensor_img.to(kUInt8));


	cv::imshow("image", final_img);
	cv::waitKey(0);
	cv::destroyAllWindows();*/
	//cv::imwrite("lung_seg.jpg", final_img);

	//Save the converted tensor into a raw file
	//writeRawData3D_v2(vec_output, "..\\sample_data\\image_b\\[fake]half_case24_256.raw");

	/*auto denorm_pred_mask = torch::where(denorm_pred[0][0] < threshold, torch::zeros_like(denorm_pred), torch::ones_like(denorm_pred));
	PRINT("denorm_pred mask min:" << denorm_pred_mask.min());
	PRINT("denorm_pred mask max:" << denorm_pred_mask.max());
	PRINT("denorm_pred_mask shape " << denorm_pred_mask.sizes());


	auto connected_lung = torch::where(denorm_pred_mask, denorm_pred[0][0], torch::full_like(denorm_pred, replace_value));
	PRINT("connected lung shape " << connected_lung.sizes());*/
	
	// --------------------------------------------------- PATCH START ----------------------------------------------------
	//int64_t patch_size = 96;
	//int64_t stride = 96;

	//PatchData patch_data = extractPatches(t_data, &normalizeTorch, NormMethod::ZSCORE, patch_size, stride);
	//Tensor patches = patch_data.patches;
	//vector<int64_t> pad_size = patch_data.pad_size;
	//vector<int64_t> v_unfold_shape = patch_data.out_shape;
	//
	//Tensor combined_patches = torch::zeros({ (long long)patches.sizes()[0], patch_size, patch_size, patch_size });

	//auto start_time = chrono::high_resolution_clock::now();
	//for (int i = 0; i < patches.sizes()[0]; i++) {
	//	PRINT("Processing patch number: " << i)
	//	torch::NoGradGuard no_grad;

	//	if (IS_HALF_PRECISION) {
	//		input = patches[i].unsqueeze(0).unsqueeze(0).to(device).to(kHalf);
	//	}
	//	else {
	//		input = patches[i].unsqueeze(0).unsqueeze(0).to(device).to(kFloat); // The defual kFloat is 32 bits
	//	}

	//	//Execute the model and turn its output into a tensor.
	//	auto output_patch = module.forward({ input }); // ouput= {1, 25, 96, 96, 96}
	//	auto output_patch_tensor = output_patch.toTensor(); // ouput= {1, 25, 96, 96, 96}
	//	output_patch_tensor = IS_HALF_PRECISION ? softmax(output_patch_tensor, 1, kHalf) : softmax(output_patch_tensor, 1, kFloat); 
	//	combined_patches[i] = squeeze(argmax(output_patch_tensor, 1)).to(kCPU).to(kFloat); // {1, 25, 96, 96, 96} => [1, 96, 96, 96] => [96, 96, 96]
	//	
	//}
	//auto end_time = chrono::high_resolution_clock::now();
	//auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);

	//print the performance in milliseconds 
	//PRINT("Inference speed in seconds: " << duration.count());
	//
	//// reconstruct the image from patches
	//Tensor recon_img = reconPatches(combined_patches, v_unfold_shape);

	// remove padding from the front of x,y,z dims.
	//Tensor seg_mask = recon_img.slice(0, pad_size[0], recon_img.sizes()[0]).slice(1, pad_size[1], recon_img.sizes()[1]).slice(2, pad_size[2], recon_img.sizes()[2]);
	//seg_mask = seg_mask.permute({2, 1, 0});
	// ----------------------------------------------- PATCH END ----------------------------------------------------------
	//Convert tensor into a 3D vector
	//vector<vector<vector<int16_t>>> vec_output = convertTensorTo3DVectorInt16(seg_mask);
	// Save the converted tensor into a raw file
	//writeRawData3D_v2(vec_output, "..\\sample_data\\image_b\\[fake]half_case24_256.raw");
	// --------------------------------------------------------- TISEPX MODEL INPUT ---------------------------------------

	auto dum_inst = torch::rand(1);
	auto dum_img = torch::rand(1);

	torch::jit::IValue in_dum1;
	torch::jit::IValue in_dum2;

	//if (IS_HALF_PRECISION) {
	//	input = t_data.unsqueeze(0).to(device).to(kHalf);
	//}
	//else {
	//	input = t_data.unsqueeze(0).to(device).to(kFloat); // The defual kFloat is 32 bits
	//}
	//auto output = module.forward({ input }); // ouput= {1, 1, 2048, 2048}
	//auto output_patch_tensor = output.toTensor(); // ouput= {1, 25, 96, 96, 96}
	//output_patch_tensor = IS_HALF_PRECISION ? softmax(output_patch_tensor, 1, kHalf) : softmax(output_patch_tensor, 1, kFloat); 
	
	INFO("Done!");

	return 0;

#endif

}