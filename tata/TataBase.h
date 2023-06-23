#pragma once

#ifdef TATA_EXPORTS
#define TATA_API __declspec(dllexport)
#else
#define TATA_API __declspec(dllimport)
#endif

// System includes
#include<iostream>
#include<fstream>
#include<cmath>
#include<vector>
#include<string>
#include<map>

using namespace std;


enum class SegTask : unsigned int {
	LUNG_SEGMENTATION,			// 0
	HEART_SEGMENTATION,			// 1
	AORTA_SEGMENTATION,			// 2
	BONE_SEGMENTATION,			// 3
	EMPHYSEMA_SEGMENTATION,		// 4
	VASCULAR_SEGMENTATION,		// 5
	TB_SEGMENTATION,			// 6
	NTM_SEGMENTATION,			// 7
	COVID_SEGMENTATION,			// 8
};


enum class RegTask : unsigned int {
	LUNG_VOLUME,				// 0
	HEART_VOLUME,				// 1
	VESSEL_VOLUME,				// 2
	AORTA_VOLUME,				// 3
};

namespace tata {


	struct PatientInfo {
		int age;
		string sex;
	};

	struct Bucket						// bucket for input and output data
	{
		vector<float> v_data;
		int64_t depth, width, height;   // depth, width, height of the input scan
		vector<float> v_pixel_spacing;
		string age;
		string sex;
		float area;
		Bucket() {
			v_data = { 0.0f, 0.0f, 0.0f };
			depth = 0;
			width = 0;
			height = 0;
			v_pixel_spacing = { 0.0f, 0.0f };
			age = "Unknown";
			sex = "Unknown";
			area = 0.0f;
		}
	};

	struct AIModel {
		
		string weight_name;
		bool half_precision;
		
		AIModel() {
			weight_name = "";
			half_precision = false;		// default is full precision (float 32)	
		}
		
	};

	struct Configuration {
		string  weight_path;
		AIModel Preprocessing;		// in case there is a preprocessing by model
		AIModel LungSegmentation;
		AIModel Heart;
		AIModel LungRegression;
		AIModel TB;
		AIModel NTM;
		AIModel COVID;
		AIModel BONE;
		AIModel AORTA;
		AIModel EMPHYSEMA;
		AIModel VASCULAR;
	};

	class TATA_API TataBase
	{
	public:
		// constructors 
		TataBase(const Configuration& config);
		// destructor
		~TataBase();

		// setters
		//void setConfiguration(const Configuration& config);

	protected:
		Configuration				m_config;
	};


	// temporary 
	class TATA_API Segmentation : public TataBase {
	public:
		Segmentation(const Configuration& config);
		~Segmentation();
		Bucket run(Bucket& bucket, SegTask seg_task);
	};

	class TATA_API Regression: public TataBase {
	public:
		Regression(const Configuration& config);
		~Regression();
		Bucket run(Bucket& bucket, RegTask reg_task);
	};

}// namespace tata