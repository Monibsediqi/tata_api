//local includes
#include "TataBase.h"
#include "utils/Preprocessing.h"
#include "utils/Postprocessing.h"
#include "utils/Optimization.h"
#include "Regression.h"


//third party includes
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>

using namespace std;
using namespace torch;
using namespace tata;

#define PRINT(x) cout << x << endl;
#define INFO(x) cout << "INFO: " << x << endl;
#define ERR(x) cerr << "ERROR: " << x << endl;


Regression::Regression(const tata::Configuration& config) {
	m_config = config;
}
Regression::~Regression() {
};

void Regression::setInputImage(const vector<float>& v_input_image) {
	m_v_input_image = v_input_image;
};

void Regression::setInputImgSizeX(const short& size_x) {
	m_input_size_x = size_x;
};

void Regression::setInputImgSizeY(const short& size_y) {
	m_input_size_y = size_y;
};

void Regression::setInputImgSizeZ(const short& size_z) {
	m_input_size_z = size_z;
};

void Regression::setPatientAge(const short& age) {
	m_patient_age = age;
};

void Regression::setPatientSex(const string& sex) {
	m_patient_sex = sex;
}


bool Regression::run(RegTask reg_task) {

	// clear gpu memory cache 
	// clear_cache();
	// check input_data;
	// checkData();
	// input data shape
	PRINT("Regression input_data size: " << m_v_input_image.size());

	vector<float> v_input = m_v_input_image;
	Tensor t_data = torch::from_blob(v_input.data(), IntArrayRef{ m_input_size_z, m_input_size_y, m_input_size_x}, kFloat32);

	INFO("start regression prediction...")
	
	Device device(torch::kCPU);

	/*if (torch::cuda::is_available())
	{
		device = torch::kCUDA;
		PRINT("Using GPU for regression");
	}
	else
	{
		device = torch::kCPU;
		PRINT("Using CPU for regression");
	}*/

	// optimize infernece of torch::jit
	//setInferenceMode();

	device = torch::kCPU;
	

	if (reg_task == RegTask::LUNG_VOLUME) {
		jit::script::Module module = jit::load(m_config.weight_path + m_config.lung_regression.weight_name, device);
		
		PRINT("Regression: t_data size: " << t_data.sizes());

		int target_size = static_cast<int>(1187.5878547836917);
		auto input_img = resizeKeepRatioXray(t_data, target_size);
		PRINT("Regression input_img min " << torch::min(input_img));
		PRINT("Regression input_img max " << torch::max(input_img));
		auto input_img_padded = padImageXray(input_img, 2048);
		PRINT("Regression input_img_padded shape  :" << input_img_padded.sizes());
		PRINT("Regression input_img_padded min " << torch::min(input_img_padded));
		PRINT("Regression input_img_padded max " << torch::max(input_img_padded));


		// P3 (Priority 3)
		auto minHU = -1025.0;		
		auto maxHU = -775.0;		

		input_img_padded = torch::clip(input_img_padded, minHU, maxHU);
		//print min max
		/*PRINT("after clip input_img_padded min " << torch::min(input_img_padded));
		PRINT("after clip input_img_padded max " << torch::max(input_img_padded));*/

		// input norm
		input_img_padded = (input_img_padded - minHU) / (maxHU - minHU);
		//print min max
		/*PRINT("after norm input_img_padded min " << torch::min(input_img_padded));
		PRINT("after norm input_img_padded max " << torch::max(input_img_padded));*/

		//auto sex = 1.0;
		float sex = m_patient_sex.compare("M") == 0 ? 1.0 : 0.0;
		PRINT("Sex " << sex)
		auto age = m_patient_age / 100;
		auto lung_area = m_area;

		float input_values[] = { lung_area, sex, age };
		int64_t num_elements = sizeof(input_values) / sizeof(float);

		auto input_feat = torch::from_blob(input_values, { 1, num_elements }, kFloat).clone().to(device);

		input_img_padded = input_img_padded.unsqueeze(0).to(device);
		//print tensor shape
		//PRINT("input_feat shape: " << input_feat.sizes()); // expecting 1x3
		//PRINT("padded_img shape: " << input_img_padded.sizes()); // // expecting [1,1,2048,2048]

		torch::jit::IValue inp_feat = input_feat;
		torch::jit::IValue in_img = input_img_padded;

		torch::NoGradGuard nograd;
		auto reg_pred = module.forward({in_img, inp_feat}).toTensor().to(kCPU);

		m_vol =  reg_pred.item<short>();
		return true;
	}
	else {
		INFO("No regression task found!");
		return false;
	}

}

short Regression::getInputImgSizeX() {
	return m_input_size_x;
};

short Regression::getInputImgSizeY() {
	return m_input_size_y;
};

short Regression::getInputImgSizeZ() {
	return m_input_size_z;
};
short Regression::getPatientAge() {
	return m_patient_age;
};

short Regression::getVolume() {
	return m_vol;
}



//string Regression::enumToString(RegTask reg_task) {
//	switch (reg_task) {
//	case RegTask::LUNG_VOLUME:
//		return "LUNG_VOLUME";
//	case RegTask::VESSEL_VOLUME:
//		return "VESSEL_VOLUME";
//	default:
//		return "UNKNOWN";
//	}
//}

//void Regression::run() {
	//setDevice();
	//loadModel();
	//setNormMethod();
	//checkData();
	//predict();
//}
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