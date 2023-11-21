#pragma once

#ifdef SEGCORE_EXPORTS
	#define SEGCORE_API __declspec(dllexport)
#else
	#define SEGCORE_API __declspec(dllimport)
#endif

// System includes
#include<iostream>
#include<fstream>
#include<cmath>
#include<vector>
#include <string>

using namespace std;

enum class Product: unsigned int
{
	MEDIP, // 0
	TISEPX, // 1
	DE_ID,	// 2
};
enum class AITask: unsigned int
{
	SEGMENTATION,  // 0
	CLASSIFICATION, // 1
	REGRESSION, // 2
	TRANSLATION, // 3
};
enum class ScanType: unsigned int {
	XRAY,			// 0
	CT,				// 1
	MR,				// 2
};
enum class NormMethod : unsigned int {
	MINMAX,			// 0
	ZSCORE,			// 1
	PERCENTILE,		// 2
	NONE,			// 3
};

struct Bucket
{
	vector<float_t> v_data;
	int64_t depth, width, height; // depth, width, height of the input scan
	vector<float> v_pixel_spacing;
	string age;
	string sex;
	ScanType scan_type;
	float area = 0;
};
//struct OutBucket
//{
//	vector <float> v_data;
//	int64_t d, w, h;
//	vector<float> v_pixel_spacing;
//	string age;
//};


class SEGCORE_API SegCoreBase
{
public :
	// constructors 
	SegCoreBase();

	// destructor
	~SegCoreBase();

	//methods
	virtual void run();

	// setters
	void setProduct(const Product& product);
	void setAITask(const AITask& ai_task);
	void setBucket(Bucket& in_bucket);


	// getters
	Product getProduct() const;
	AITask getAITask() const;
	Bucket getBucket() const;
	//OutBucket getOutBucket() const;

protected:

	//methods
	virtual void setNormMethod(); // set norm based on task
	
	//objects	
	NormMethod			m_norm_method;

private:
	
	Product				m_product;
	AITask				m_ai_task;
	Bucket				m_bucket;
	//OutBucket			m_out_bucket;

};
//////////////////////////// Segmentation /////////////////////////
//struct ModelName {
//	string lung = "xray2lung.mipx";
//	string vessels = "vessels.mipx";
//};

enum class SegTask {
	// MEDIP Segmentation Tasks
	// ViT Grand Segmentation
	CARDIAC,
	MUSCLES,
	VERTEBRAE,
	RIB,
	OTHER_ORGANS,
	ALL_104,

	// DEEP CATCH
	DEEPCATCH,

	// TISEPX
	LUNG,
	VESSELS,
	AORTA,
};
 class SEGCORE_API Segmentation : public SegCoreBase
{
public:
	//TODO: Fixe the device issue
	Segmentation();
	~Segmentation();

	// Inherited via SegCoreBase
	virtual void run() override;

	void setSegTask(SegTask seg_task);
	void setModelPath(string model_path);
	// handle thread in here
	Bucket predict(Bucket);

	// getters
	string getModel();
	SegTask getSegTask();

private:
	
	std::string enumToString(SegTask seg_task);
	void setModule();
	void setNormMethod() override;
	void setDevice();
	
	void checkData();		// At the moment it checks whether the data is xray or CT. 
							// In the future it will check whether the data is valid or not.
	bool				m_isXray;
	bool				m_isCT; 

	//ModelName			m_model_name;  // struct
	SegTask				m_seg_task; // enum class
	string				m_model_path;

};

 enum class RegTask : unsigned int {
	 // MEDIP Regression Tasks

	 // TISEPX Regression Tasks
	 LUNG_VOLUME,				// 0
	 VESSEL_VOLUME,				// 1
	 AORTA_VOLUME,				// 2
 };

 /////////////////////////// Regression /////////////////////////
 class SEGCORE_API Regression : public SegCoreBase
 {
public:
	 Regression();
	 ~Regression();

	 // inherited via SegCoreBase
	 virtual void run() override;

	 // handle thread in here
	 Bucket predict(Bucket);

	 //settters
	 void setRegTask(RegTask reg_task);
	 void setModelPath(string model_path);

	 // getters
	 string getModelPath();

 private:
	 std::string enumToString(RegTask seg_task);
	 void setNormMethod() override;
	 string				m_model_path;  // struct
	 RegTask			m_reg_task; // enum class
	 NormMethod 	    m_norm_method;
 };
