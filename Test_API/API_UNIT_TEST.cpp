// Test_API.cpp : This file contains the 'main' function. Program execution begins and ends there.


#include <iostream>
//#include "SegCoreBase.h"
#include "TataBase.h"

using namespace std;
using namespace tata;

#define PRINT(x) std::cout << x << std::endl;
#define INFO(x) std::cout << "INFO: " << x << std::endl;
#define ERR(x) std::cerr << "ERROR: " << x << std::endl;

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

//void testXrayLungSeg() {
//
//
//	// Models path
//	const std::string	XRAY2LUNG = "F:\\2023\\AI\\App\\scripted_models\\xray2lung.mipx";
//	const std::string	LUNG_REG = "F:\\2023\\AI\\App\\scripted_models\\lungregression.mipx";
//
//
//	string LUNG_VOL_RAW = "F:\\2023\\AI\\sample_data\\tisepx_a\\lung_vol_raw\\JB0006_CXR_0base_201229.raw";
//
//	int64_t XRAY_D = 1;
//	int64_t XRAY_W = 2652;
//	int64_t XRAY_H = 2450;
//
//
//	vector<short> vec_img_A = readRawFile1D(LUNG_VOL_RAW, XRAY_D, XRAY_W, XRAY_H); // read image A
//	vector<float_t> vec_img_A_float(vec_img_A.begin(), vec_img_A.end());
//
//	// start input bucket setup 
//	Bucket bucket;
//
//	bucket.v_data = vec_img_A_float;			// type float vector 
//	bucket.age = "30";							// type string
//	bucket.depth = XRAY_D;						// type int64_t
//	bucket.width = XRAY_W;						// type int64_t
//	bucket.height = XRAY_H;						// type int64_t
//	bucket.v_pixel_spacing = { 0.12, 0.12 };	// type float vector with size 2
//	bucket.scan_type = ScanType::XRAY;
//	// -- end input bucket setup -- 
//	INFO("inBucket setup successful!")
//	
//
//	// -- start API base setup --
//	SegCoreBase seg_core_base;
//	seg_core_base.setProduct(Product::TISEPX);
//	seg_core_base.setAITask(AITask::SEGMENTATION);
//	//seg_core_base.setInBucket(in_bucket);
//	 
//	// -- end API base setup --
//
//	// calling segmentation models
//	Segmentation segObj;
//	segObj.setSegTask(SegTask::LUNG);
//	segObj.setModelPath(XRAY2LUNG); // complete path to model.mipx
//
//	Bucket outbucket = segObj.predict(bucket);
//	PRINT("Done Segmentation!");
//	
//	Regression regObj;
//	regObj.setRegTask(RegTask::LUNG_VOLUME);
//	regObj.setModelPath(LUNG_REG);
//	//regObj.setBucket(outbucket);
//	Bucket regBucket = regObj.predict(outbucket);
//	PRINT("Done Regression");
//}

//void testTataGeneral() {
//	// Initialize a Fibonacci relation sequence.
//	fibonacci_init(1, 1);
//	// Write out the sequence values until overflow.
//	do {
//		std::cout << fibonacci_index() << ": "
//			<< fibonacci_current() << std::endl;
//	} while (fibonacci_next());
//	// Report count of values written before overflow.
//	std::cout << fibonacci_index() + 1 <<
//		" Fibonacci sequence values fit in an " <<
//		"unsigned 64-bit integer." << std::endl;
//
//}


//void testTataAPI() {
//
//	const std::string	XRAY2LUNG = "F:\\2023\\AI\\App\\scripted_models\\xray2lung.mipx";
//	const std::string	LUNG_REG = "F:\\2023\\AI\\App\\scripted_models\\lungregression_cpu.mipx";
//
//
//	string LUNG_VOL_RAW = "F:\\2023\\AI\\sample_data\\tisepx_a\\lung_vol_raw\\JB0006_CXR_0base_201229.raw";
//
//	int64_t XRAY_D = 1;
//	int64_t XRAY_W = 2652;
//	int64_t XRAY_H = 2450;
//
//
//	vector<short> vec_img_A = readRawFile1D(LUNG_VOL_RAW, XRAY_D, XRAY_W, XRAY_H); // read image A
//	vector<float_t> vec_img_A_float(vec_img_A.begin(), vec_img_A.end());
//
//	// start input bucket setup 
//	Bucket bucket;
//
//	bucket.v_data = vec_img_A_float;			// type float vector 
//	bucket.age = "30";							// type string
//	bucket.sex = "M";
//	bucket.depth = XRAY_D;						// type int64_t
//	bucket.width = XRAY_W;						// type int64_t
//	bucket.height = XRAY_H;						// type int64_t
//	bucket.v_pixel_spacing = { 0.12, 0.12 };	// type float vector with size 2
//
//	// -- end input bucket setup -- 
//	INFO("Bucket setup successful!")
//
//
//	// -- start API base setup --
//	TataBase tata_base;
//	tata_base.setAITask(AITask::SEGMENTATION);
//	//seg_core_base.setInBucket(in_bucket);
//
//	// -- end API base setup --
//
//	// calling segmentation models
//	Segmentation segObj;
//	segObj.setSegTask(SegTask::LUNG);
//	segObj.setModelPath(XRAY2LUNG); // complete path to model.mipx
//
//	Bucket outbucket = segObj.predict(bucket);
//	PRINT("Done Segmentation!");
//
//	Regression regObj;
//	regObj.setRegTask(RegTask::LUNG_VOLUME);
//	regObj.setModelPath(LUNG_REG);
//	//regObj.setBucket(outbucket);
//	Bucket regBucket = regObj.predict(outbucket);
//	PRINT("Done Regression");
//
//}

void tataAPITest2() {
	string LUNG_VOL_RAW = "F:\\2023\\AI\\sample_data\\tisepx_a\\lung_vol_raw\\JB0006_CXR_0base_201229.raw";

	int64_t XRAY_D = 1;
	int64_t XRAY_W = 2652;
	int64_t XRAY_H = 2450;


	vector<short> vec_img_A = readRawFile1D(LUNG_VOL_RAW, XRAY_D, XRAY_W, XRAY_H); // read image A
	vector<float_t> vec_img_A_float(vec_img_A.begin(), vec_img_A.end());

	// start input bucket setup 
	Bucket bucket;

	bucket.v_data = vec_img_A_float;			// type float vector 
	bucket.age = "30";							// type string
	bucket.sex = "M";
	bucket.depth = XRAY_D;						// type int64_t
	bucket.width = XRAY_W;						// type int64_t
	bucket.height = XRAY_H;						// type int64_t
	bucket.v_pixel_spacing = { 0.12, 0.12 };	// type float vector with size 2

	// -- end input bucket setup -- 
	INFO("Bucket setup successful!")

	const string	weight_path = "F:\\2023\\AI\\App\\scripted_models\\";
	const string	xray2lung_weight = "xray2lung.mipx";
	const string	lung_reg_weight = "lungregression_cpu.mipx";

	PRINT(weight_path + xray2lung_weight)

	Configuration model_config;
	model_config.weight_path = weight_path;
	model_config.LungSegmentation.weight_name = xray2lung_weight;
	model_config.LungRegression.weight_name = lung_reg_weight;

	// -- API base setup --
	TataBase tata_api (model_config);
	//tata_api.setConfiguration(model_config);


	Segmentation segment(model_config);
	Bucket lung_seg_bucket = segment.run(bucket, SegTask::LUNG_SEGMENTATION);

	INFO("Done Segmentation")

	Regression regress(model_config);
	Bucket lungRegBucket = regress.run(lung_seg_bucket, RegTask::LUNG_VOLUME);

	INFO("Done Regression");
}

int main()
{
	//testXrayLungSeg();
	//testTataGeneral();
	tataAPITest2();

	

	INFO("Test completed!")
}
