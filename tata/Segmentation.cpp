// local includes
#include "utils/Preprocessing.h"
#include "TataBase.h"

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


Segmentation::Segmentation(const Configuration& config) : TataBase(config)
{
}
Segmentation::~Segmentation()
{
}
Bucket Segmentation::run(Bucket& bucket, SegTask seg_task)
{
	// do something in here
	
	//string model_path = m_config.weight_path + weight_name;
	PRINT("model name: " << m_config.LungSegmentation.weight_name);

	Device device(torch::kCUDA);
	//FIXME: setup jit configuration

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


	// set segmentation module
	//setModule();
	
	
	Bucket bucket_out = Bucket();

	if (seg_task == SegTask::LUNG_SEGMENTATION) {
		string model_path = m_config.weight_path + m_config.LungSegmentation.weight_name;
		PRINT("Segmentation model path: " << m_config.weight_path + m_config.LungSegmentation.weight_name);
		jit::script::Module module = jit::load(model_path, device);
		Tensor t_data = torch::from_blob(bucket.v_data.data(), IntArrayRef{ bucket.depth, bucket.height, bucket.width }, kFloat32);
		// assert the the depth is 1

		t_data = resizeKeepRatioXray(t_data, 2048);
		t_data = padImageXray(t_data, 2048, 0);
		PRINT("Check padded t_data sizes: " << t_data.sizes());
		t_data = normalizeTorch(t_data, NormMethod::PERCENTILE);
		PRINT("normalized t_data min max: " << t_data.min() << t_data.max());

		jit::IValue input = t_data.unsqueeze(0).to(device);

		torch::NoGradGuard no_grad;
		auto output = module.forward({ input }); // {1,1,2048,2048}
		auto t_output = output.toTensor().to(kCPU);
		auto lung_seg = t_output.squeeze(0);
		PRINT("lun seg shape: " << lung_seg.sizes());

		//TODO: handle these values
		int b_min_value = -1100;
		int b_max_value = -500;
		lung_seg = lung_seg * (b_max_value - b_min_value) + b_min_value;

		// apply threshold 
		float threshold = -1015.0;
		float replace_value = -1024;
		lung_seg = torch::where(lung_seg[0] < threshold, torch::full_like(lung_seg, replace_value), lung_seg);
		auto lung_seg_mask = torch::where(lung_seg[0] < threshold, torch::zeros_like(lung_seg[0]), torch::ones_like(lung_seg[0]));

		auto pixel_size_resize_w = 0.18517382812499997;
		auto pixel_size_resize_h = 0.18517382812499997;

		//get mask area
		auto area = torch::sum(lung_seg_mask);
		area = area * pixel_size_resize_w * pixel_size_resize_h / 100; // equivalent 
		PRINT("lung area: " << area);

		// TODO: create type neutral bucket
		vector<float> v_data(lung_seg.data_ptr<float>(), lung_seg.data_ptr<float>() + lung_seg.numel());
		float f_area = area.item<float>();

		

		bucket_out.v_data = v_data;
		bucket_out.depth = lung_seg.size(0);
		bucket_out.height = lung_seg.size(1);
		bucket_out.width = lung_seg.size(2);
		bucket_out.age = bucket.age;
		bucket_out.sex = bucket.sex;
		bucket_out.area = f_area;
		return bucket_out;
	}
	else {
	
		PRINT("No segmentation is performed, returning input bucket");
		return bucket;
	}
}

//void Segmentation::setSegTask(SegTask task)
//{
//	m_seg_task = task;
//}
//Bucket Segmentation::predict(Bucket in_bucket)
//{
//
//	//checkData();
//	m_isXray = true;
//	Device device(torch::kCUDA);
//	//FIXME: setup jit configuration
//
//	//TODO: check input for scan type
//	if (torch::cuda::is_available())
//	{
//		device = torch::kCUDA;
//		PRINT("Using GPU for segmentation");
//	}
//	else
//	{
//		device = torch::kCPU;
//		PRINT("Using CPU for segmentation");
//	}
//
//
//	// set segmentation module
//	//setModule();
//	std::string model_path = this->m_model_path;
//	PRINT("Segmentation model path: " << m_model_path);
//	jit::script::Module module = jit::load(model_path, device);
//	Tensor t_data = torch::from_blob(in_bucket.v_data.data(), IntArrayRef{ in_bucket.depth,in_bucket.height, in_bucket.width }, kFloat32);
//	
//	Bucket bucket;
//
//	if (m_seg_task == SegTask::LUNG)
//	
//	{
//		if (m_isXray)
//		{
//			// assert the the depth is 1
//
//			t_data = resizeKeepRatioXray(t_data, 2048);
//			t_data = padImageXray(t_data, 2048, 0);
//			PRINT("Check padded t_data sizes: " << t_data.sizes());
//			t_data = normalizeTorch(t_data, NormMethod::PERCENTILE);
//			PRINT("normalized t_data min max: " << t_data.min() << t_data.max());
//
//			jit::IValue input = t_data.unsqueeze(0).to(device);
//
//			torch::NoGradGuard no_grad;
//			auto output = module.forward({ input }); // {1,1,2048,2048}
//			auto t_output = output.toTensor().to(kCPU);
//			auto lung_seg = t_output.squeeze(0);
//			PRINT("lun seg shape: " << lung_seg.sizes());
//
//			//TODO: handle these values
//			int b_min_value = -1100;
//			int b_max_value = -500;
//			lung_seg = lung_seg * (b_max_value - b_min_value) + b_min_value;
//
//			// apply threshold 
//			float threshold = -1015.0;
//			float replace_value = -1024;
//			lung_seg = torch::where(lung_seg[0] < threshold, torch::full_like(lung_seg, replace_value), lung_seg);
//			auto lung_seg_mask = torch::where(lung_seg[0] < threshold, torch::zeros_like(lung_seg[0]), torch::ones_like(lung_seg[0]));
//
//			auto pixel_size_resize_w = 0.18517382812499997;
//			auto pixel_size_resize_h = 0.18517382812499997;
//
//			//get mask area
//			auto area = torch::sum(lung_seg_mask);
//			area = area * pixel_size_resize_w * pixel_size_resize_h / 100; // equivalent 
//			PRINT("lung area: " << area);
//
//
//
//			// TODO: create type neutral bucket
//			vector<float_t> v_data(lung_seg.data_ptr<float>(), lung_seg.data_ptr<float>() + lung_seg.numel());
//			float f_area = area.item<float>();
//
//			
//
//			bucket.v_data = v_data;
//			bucket.depth = lung_seg.size(0);
//			bucket.height = lung_seg.size(1);
//			bucket.width = lung_seg.size(2);
//			bucket.age = in_bucket.age;
//			bucket.sex = in_bucket.sex;
//			bucket.area = f_area;
//
//			return bucket;
//		}
//	}
//	else {
//		return in_bucket;
//		PRINT("Returning input bucket back!")
//	}
//}
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
