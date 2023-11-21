// local includes
#include "utils/Preprocessing.h"
#include "utils/Postprocessing.h"
#include "utils/Optimization.h"
#include "Segmentation.h"

// third party includes
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


Segmentation::Segmentation(const Configuration& config) 
{
	m_config = config;
}
Segmentation::~Segmentation()
{
}
bool Segmentation::run(SegTask seg_task){

	// do something in here
	vector<float> v_p_data = convertShortToFloatVector(m_v_input_image);

	Tensor t_data = torch::from_blob(v_p_data.data(), IntArrayRef{m_input_size_z, m_input_size_y, m_input_size_x}, kFloat32);
	// input data min-max
	PRINT("run:: input min-max: " << t_data.min() << " " << t_data.max());
	Device device(torch::kCUDA);

	//TODO: check input for scan type
	if (torch::cuda::is_available())
	{
		device = torch::kCUDA;
		PRINT("Using GPU for segmentation");
	}
	else
	{
		device = torch::kCPU;
		PRINT("Using CPU for segmentation");
	}

	// optimizing infernece speed of torch::jit
	setInferenceMode();

	torch::NoGradGuard no_grad;

	if (seg_task == SegTask::LUNG_SEGMENTATION) {
		
		PRINT("Lung Segmentation model path: " << m_config.weight_path + m_config.lung_segmentation.weight_name);
		jit::script::Module module = jit::load(m_config.weight_path + m_config.lung_segmentation.weight_name, device);
		
		t_data = resizeKeepRatioXray(t_data, 2048);
		t_data = padImageXray(t_data, 2048, 0);
		PreprocessedData preprocessed_data = normalizeTorch(t_data, NormMethod::PERCENTILE);

		jit::IValue input = preprocessed_data.t_data.unsqueeze(0).to(device);


		auto output = module.forward({ input });				// {1,1,2048,2048}
		auto t_output = output.toTensor().to(kCPU);
		auto lung_seg = t_output.squeeze(0);
		PRINT("lun seg shape: " << lung_seg.sizes());
		PRINT("Min-max: " << lung_seg.min() << " " << lung_seg.max());


		// Fixed values based on python code
		int b_min_value = -1100;
		int b_max_value = -500;
		lung_seg = lung_seg * (b_max_value - b_min_value) + b_min_value;

		// apply threshold. Fixed values based on python code
		float threshold = -1015.0;
		float replace_value = -1024;
		lung_seg = torch::where(lung_seg[0] < threshold, torch::full_like(lung_seg, replace_value), lung_seg);
		auto lung_seg_mask = torch::where(lung_seg[0] < threshold, torch::zeros_like(lung_seg[0]), torch::ones_like(lung_seg[0]));
		// print min max of the lung_seg
		PRINT("lung_seg min max: " << lung_seg.min() << " " << lung_seg.max());



		// TODO: Apply getConnectedRegion function in here
		auto ratio = float(2048) / max(m_input_size_x, m_input_size_y);
		auto pixel_size_resize_w = m_img_spacing_x / ratio;
		auto pixel_size_resize_h = m_img_spacing_y / ratio;
		PRINT("Ratio: " << ratio);
		

		//get mask area
		auto area = torch::sum(lung_seg_mask.flatten()).item<float>();
		area = area * pixel_size_resize_w * pixel_size_resize_h / 100;

		//print lung seg size
		PRINT("lung seg size:" << lung_seg.sizes())

		vector<float> v_data(lung_seg.data_ptr<float>(), lung_seg.data_ptr<float>() + lung_seg.numel());
		
		m_v_output_image = v_data;
		m_area = area;
		m_output_size_z = lung_seg.sizes().vec()[0]; // z
		m_output_size_y = lung_seg.sizes().vec()[1]; // y
		m_output_size_x = lung_seg.sizes().vec()[2]; // x
		return true;
	}
	if (seg_task == SegTask::COVID_SEGMENTATION) {
		PRINT("COVID Segmentation model path: " << m_config.weight_path + m_config.COVID.weight_name);
		jit::script::Module module = jit::load(m_config.weight_path + m_config.COVID.weight_name, device);		
		/*PRINT("t_data shape: " << t_data.sizes());
		PRINT("min max before resize: " << t_data.min() << t_data.max());*/
		auto a_min_val = -1100;
		auto a_max_val = -500;

		t_data = resizeKeepRatioXray(t_data, 2048);
		//PRINT("min max after resize: " << t_data.min() << t_data.max());
		t_data = padImageXray(t_data, 2048, a_min_val);

		
		// normalization
		float eps = 1e-10;
		t_data = (t_data - a_min_val) / ((a_max_val - a_min_val) + eps);
		PRINT("COVID normalized t_data min max: " << t_data.min() << t_data.max());

		jit::IValue input = t_data.unsqueeze(0).to(device);

		auto output = module.forward({ input });				// {1,1, 2048, 2048}
		auto t_output = output.toTensor().to(kCPU);
		auto covid_seg = t_output.squeeze(0);
		//PRINT("lun seg shape: " << lung_seg.sizes());

		// Denormalization: Fixed values based on python code
		int b_min_value = -1100;
		int b_max_value = -400;
		covid_seg = covid_seg * (b_max_value - b_min_value) + b_min_value;
		//PRINT("Denorm min max: " << covid_seg.min() << covid_seg.max());

		// apply threshold. Fixed values based on python code
		float threshold = -950.0;
		float replace_value = -1024;
		covid_seg = torch::where(covid_seg[0] < threshold, torch::full_like(covid_seg, replace_value), covid_seg);
		auto covid_seg_mask = torch::where(covid_seg[0] < threshold, torch::zeros_like(covid_seg[0]), torch::ones_like(covid_seg[0]));

		auto ratio = float(2048) / max(m_input_size_x, m_input_size_y);

		auto pixel_size_resize_w = m_img_spacing_x / ratio;
		auto pixel_size_resize_h = m_img_spacing_y / ratio;

		//get mask area
		auto area = torch::sum(covid_seg_mask.flatten()).item<float>();
		/*PRINT("area before mul : "<< area)*/
		area = area * pixel_size_resize_w * pixel_size_resize_h / 100;
		//float f_area = area.item<float>();
		//PRINT("covid area: " << area);

		// TODO: create type neutral bucket
		vector<float> v_data(covid_seg.data_ptr<float>(), covid_seg.data_ptr<float>() + covid_seg.numel());
		m_v_output_image = v_data;
		m_area = area;
		m_output_size_z = covid_seg.sizes().vec()[0]; // z
		m_output_size_y = covid_seg.sizes().vec()[1]; // y
		m_output_size_x = covid_seg.sizes().vec()[2]; // x
		return true;

	}
	if (seg_task == SegTask::HEART_SEGMENTATION) {
		PRINT("Heart Segmentation model path: " << m_config.weight_path + m_config.heart_segmentation.weight_name);
		jit::script::Module module = jit::load(m_config.weight_path + m_config.heart_segmentation.weight_name, device);
		
		t_data = resizeKeepRatioXray(t_data, 512);
		//PRINT("min max after resize: " << t_data.min() << t_data.max());
		t_data = padImageXray(t_data, 512, 0);

		// normalization
		PreprocessedData preprocessed_data = normalizeTorch(t_data, NormMethod::PERCENTILE);

		jit::IValue input = preprocessed_data.t_data.unsqueeze(0).to(device);;
		auto output = module.forward({ input });				// {1,1, 512, 512}
		auto t_output = output.toTensor().to(kCPU);
		auto heart_seg = t_output.squeeze(0);

		// Denormalization: Fixed values based on python code
		int b_min_value = -1100;
		int b_max_value = -500;
		heart_seg = heart_seg * (b_max_value - b_min_value) + b_min_value;

		// apply threshold. Fixed values based on python code
		float threshold = -1015.0;
		float replace_value = -1024;
		heart_seg = torch::where(heart_seg[0] < threshold, torch::full_like(heart_seg, replace_value), heart_seg);
		auto heart_seg_mask = torch::where(heart_seg[0] < threshold, torch::zeros_like(heart_seg[0]), torch::ones_like(heart_seg[0]));

		auto ratio = float(512) / max(m_input_size_x, m_input_size_y);

		auto pixel_size_resize_w = m_img_spacing_x / ratio;
		auto pixel_size_resize_h = m_img_spacing_y / ratio;

		//get mask area
		auto area = torch::sum(heart_seg_mask.flatten()).item<float>();
		//PRINT("area before mul : "<< area)
		area = area * pixel_size_resize_w * pixel_size_resize_h / 100;

		// TODO: create type neutral bucket
		vector<float> v_data(heart_seg.data_ptr<float>(), heart_seg.data_ptr<float>() + heart_seg.numel());

		m_v_output_image = v_data;
		m_area = area;
		m_output_size_z = heart_seg.sizes().vec()[0]; // z
		m_output_size_y = heart_seg.sizes().vec()[1]; // y
		m_output_size_x = heart_seg.sizes().vec()[2]; // x
		return true;

	}
	if (seg_task == SegTask::VASCULAR_SEGMENTATION) {
		PRINT("Vascualr Segmentation model path: " << m_config.weight_path + m_config.vessel_segmentation.weight_name);
		jit::script::Module module = jit::load(m_config.weight_path + m_config.vessel_segmentation.weight_name, device);
		auto a_min_val = -1100;
		auto a_max_val = -500;

		t_data = resizeKeepRatioXray(t_data, 2048);
		t_data = padImageXray(t_data, 2048, a_min_val);

		// normalization
		float eps = 1e-10;
		t_data = (t_data - a_min_val) / ((a_max_val - a_min_val) + eps);

		jit::IValue input = t_data.unsqueeze(0).to(device);

		auto output = module.forward({ input });				// {1,1, 2048, 2048}
		auto t_output = output.toTensor().to(kCPU);
		auto vascular_seg = t_output.squeeze(0);

		// Denormalization: Fixed values based on python code
		int b_min_value = -1100;
		int b_max_value = -500;
		vascular_seg = vascular_seg * (b_max_value - b_min_value) + b_min_value;
		PRINT("Denorm min max: " << vascular_seg.min() << vascular_seg.max());

		// apply threshold. Fixed values based on python code
		float threshold = -1015.0;
		float replace_value = -1024;
		vascular_seg = torch::where(vascular_seg[0] < threshold, torch::full_like(vascular_seg, replace_value), vascular_seg);
		auto vascular_seg_mask = torch::where(vascular_seg[0] < threshold, torch::zeros_like(vascular_seg[0]), torch::ones_like(vascular_seg[0]));

		auto ratio = float(2048) / max(m_input_size_x, m_input_size_y);
		auto pixel_size_resize_w = m_img_spacing_x / ratio;
		auto pixel_size_resize_h = m_img_spacing_y / ratio;

		//get mask area
		auto area = torch::sum(vascular_seg_mask.flatten()).item<float>();
		area = area * pixel_size_resize_w * pixel_size_resize_h / 100;

		vector<float> v_data(vascular_seg.data_ptr<float>(), vascular_seg.data_ptr<float>() + vascular_seg.numel());
		m_v_output_image = v_data;
		m_area = area;
		m_output_size_z = vascular_seg.sizes().vec()[0]; // z
		m_output_size_y = vascular_seg.sizes().vec()[1]; // y
		m_output_size_x = vascular_seg.sizes().vec()[2]; // x
		return true;
	}
	if (seg_task == SegTask::BONE_SEGMENTATION) {
		jit::script::Module module = jit::load(m_config.weight_path + m_config.bone_segmentation.weight_name, device);
		t_data = resizeKeepRatioXray(t_data, 2048);
		t_data = padImageXray(t_data, 2048, 0);
		PreprocessedData preprocessed_data = normalizeTorch(t_data, NormMethod::ZSCORE);

		jit::IValue input = preprocessed_data.t_data.unsqueeze(0).to(device);

		auto output = module.forward({ input });				// {1,1,2048,2048}
		auto t_output = output.toTensor().to(kCPU);
		auto bone_seg = t_output.squeeze(0);

		bone_seg = bone_seg * preprocessed_data.norm_params.std + preprocessed_data.norm_params.avg;		// Denormalization: Fixed values based on python code
		vector<float> v_data(bone_seg.data_ptr<float>(), bone_seg.data_ptr<float>() + bone_seg.numel());
		
		//setup output
		m_v_output_image = v_data;
		m_output_size_z = bone_seg.sizes().vec()[0]; // z
		m_output_size_y = bone_seg.sizes().vec()[1]; // y
		m_output_size_x = bone_seg.sizes().vec()[2]; // x
		return true;
		
	}
	else {
		PRINT("No segmentation is performed, returning input data");
		m_v_output_image = v_p_data;
		m_area = 0.0;
		return false;
	}
}

//setters
void Segmentation::setInputImage(vector<short> v_input_image) {
	m_v_input_image = v_input_image;
}
void Segmentation::setInputSizeX(short size_x) {
	m_input_size_x = size_x; 
}

void Segmentation::setInputSizeY(short size_y) {
	m_input_size_y = size_y;
};
void Segmentation::setInputSizeZ(short size_z) {
	m_input_size_z = size_z;
};

void Segmentation::setImgSpacingX(float img_spacing_x) {
	m_img_spacing_x = img_spacing_x; 
};
void Segmentation::setImgSpacingY(float img_spacing_y) {
	m_img_spacing_y = img_spacing_y; 
};
void Segmentation::setImgSpacingZ(float img_spacing_z) {
	m_img_spacing_z = img_spacing_z; 
};


vector<float>Segmentation::getOutputImage() {
	return m_v_output_image;
}
short Segmentation::getOutputSizeX() {
	return m_output_size_x; 
}
short Segmentation::getOutputSizeY() {
	return m_output_size_y; 
}
short Segmentation::getOutputSizeZ() {
	return m_output_size_z; 
}

float Segmentation::getImgSpacingX() {
	return m_img_spacing_x;
}
float Segmentation::getImgSpacingY() {
	return m_img_spacing_y;
}
float Segmentation::getImgSpacingZ() {
	return m_img_spacing_z;
}

float Segmentation::getArea() {
	return m_area;
}



//void Segmentation::setModelPath(string model_path) {
//	this->m_model_path = model_path;
//}
//string Segmentation::enumToString(SegTask seg_task) {
//	switch (seg_task) {
//	case SegTask::CARDIAC:
//		return "CARDIAC";
//	case SegTask::MUSCLES:
//		return "MUSCLES";
//	case SegTask::VERTEBRAE:
//		return "VERTEBRAE";
//	case SegTask::RIB:
//		return "RIB";
//	case SegTask::OTHER_ORGANS:
//		return "OTHER_ORGANS";
//	case SegTask::ALL_104:
//		return "ALL_104";
//	case SegTask::DEEPCATCH:
//		return "DEEPCATCH";
//	case SegTask::LUNG:
//		return "LUNG";
//	case SegTask::VESSELS:
//		return "VESSELS";
//	case SegTask::AORTA:
//		return "AORTA";
//	default:
//		return "Unknown";
//	}
//}
