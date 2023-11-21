//local includes
#include "../SegCoreBase.h"
#include "../utils/Postprocessing.h"
#include "../utils/Preprocessing.h"

//third party includes
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>

using namespace std;
using namespace torch;

#define PRINT(x) cout << x << endl;
#define INFO(x) cout << "INFO: " << x << endl;
#define ERR(x) cerr << "ERROR: " << x << endl;


Regression::Regression() {};
Regression::~Regression() {};

void Regression::setModelPath(string model_path) {
	this->m_model_path = model_path;
}
void Regression::setRegTask(RegTask reg_task) {
	this->m_reg_task = reg_task;
}
void Regression::setNormMethod() {
	try
	{
		// FIXME: this is temporary
		m_norm_method = NormMethod::NONE;// Using no norm
		PRINT("Using NormMethod::NONE")
		//if (m_reg_task == RegTask::LUNG_VOLUME) {
		//	m_norm_method = NormMethod::PERCENTILE; // XRay lung uses percentile norm

		//}
		//else if (m_reg_task == RegTask::VESSEL_VOLUME)
		//{
		//	m_norm_method = NormMethod::MINMAX; // CT uses min-max norm
		//}
		//else
		//{
		//	m_norm_method = NormMethod::NONE;// Using no norm
		//	PRINT("Using NormMethod::NONE")
		//}
	}
	catch (exception e)
	{
		ERR("Normalization method not found for task: " << enumToString(m_reg_task));
	}
}
Bucket Regression::predict(Bucket bucket) {

	// clear gpu memory cache 
	// clear_cache();
	// check input_data;
	// checkData();
	
	Device device(torch::kCUDA);
	if (torch::cuda::is_available())
	{
		device = torch::kCUDA;
		PRINT("Using GPU for regression");
	}
	else
	{
		device = torch::kCPU;
		PRINT("Using CPU for regression");
	}
	
	setNormMethod();
	std::string model_path = this->m_model_path;
	jit::script::Module module = jit::load(model_path, device);
	Tensor t_data = torch::from_blob(bucket.v_data.data(), IntArrayRef{ bucket.depth,bucket.height, bucket.width }, kFloat32);
	PRINT("Regression: t_data size: " << t_data.sizes());
	if (m_reg_task == RegTask::LUNG_VOLUME) {

		//Temporary 
		//convert primary type float to int


		int target_size = static_cast<int>(1187.5878547836917);
		auto input_img = resizeKeepRatioXray(t_data, target_size);
		PRINT("Regression input_img min " << torch::min(input_img));
		PRINT("Regression input_img max " << torch::max(input_img));
		auto input_img_padded = padImageXray(input_img, 2048);
		PRINT("Regression input_img_padded shape  :" << input_img_padded.sizes());
		PRINT("Regression input_img_padded min " << torch::min(input_img_padded));
		PRINT("Regression input_img_padded max " << torch::max(input_img_padded));

		auto minHU = -1025.0;
		auto maxHU = -775.0;

		input_img_padded = (input_img_padded - minHU) / (maxHU - minHU);
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
		auto lung_area = bucket.area;

		float input_values[] = { lung_area, sex, age };
		int64_t num_elements = sizeof(input_values) / sizeof(float);

		auto input_feat = torch::from_blob(input_values, { 1, num_elements }, kFloat).clone().to(device);

		input_img_padded = input_img_padded.unsqueeze(0).to(device);
		//print tensor shape
		PRINT("input_feat shape: " << input_feat.sizes()); // expecting 1x3
		PRINT("padded_img shape: " << input_img_padded.sizes()); // // expecting [1,1,2048,2048]

		PRINT("input feat " << input_feat);
		PRINT("input_img_padded " << input_img_padded[0][0][0][0]);


		torch::jit::IValue inp_feat = input_feat;
		torch::jit::IValue in_img = input_img_padded;

		torch::NoGradGuard nograd;
		auto reg_pred = module.forward({ in_img, inp_feat }).toTensor().to(kCPU);
		//print output shape
		PRINT("Regression Predicted Value: " << reg_pred);

		Bucket reg_Bucket = bucket;
		return reg_Bucket;
	}

}
string Regression::enumToString(RegTask reg_task) {
	switch (reg_task) {
	case RegTask::LUNG_VOLUME:
		return "LUNG_VOLUME";
	case RegTask::VESSEL_VOLUME:
		return "VESSEL_VOLUME";
	default:
		return "UNKNOWN";
	}
}

void Regression::run() {
	//setDevice();
	//loadModel();
	//setNormMethod();
	//checkData();
	//predict();
}
//void Regression::setDevice() {
//	if (torch::cuda::is_available()) {
//		INFO("CUDA is available! Training on GPU.");
//		m_device = torch::Device(torch::kCUDA);
//	}
//	else {
//		INFO("CUDA is not available! Training on CPU.");
//		m_device = torch::Device(torch::kCPU);
//	}
//}
//void Regression::loadModel() {
//	try {
//		// Deserialize the ScriptModule from a file using torch::jit::load().
//		m_model = torch::jit::load(m_model_path);
//	}
//	catch (const c10::Error& e) {
//		ERR("Error loading the model: " << m_model_path);
//	}
//}
//void Regression::setModel(string model_path, RegTask reg_task) {
//	this->setModelPath(model_path);
//	this->setRegTask(reg_task);
//	this->setNormMethod();
//	//this->setDevice();
//	//this->loadModel();
//}