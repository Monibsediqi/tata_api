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
#include <tuple>
#include<map>
#include<unordered_map>
#include <mutex>

using namespace std;


enum class SegTask : unsigned int {
	LUNG_SEGMENTATION,			// 0
	HEART_SEGMENTATION,			// 1
	AORTA_SEGMENTATION,			// 2
	BONE_SEGMENTATION,			// 3
	EMPHYSEMA_SEGMENTATION,		// 4
	VASCULAR_SEGMENTATION,		// 5
	AIRWAY_SEGMENTATION,		// 6
	TB_SEGMENTATION,			// 7
	NTM_SEGMENTATION,			// 8
	COVID_SEGMENTATION,			// 9
};

enum class RegTask : unsigned int {
	LUNG_VOLUME,				// 0
	HEART_VOLUME,				// 1
	VESSEL_VOLUME,				// 2
	AORTA_VOLUME,				// 3
};

enum StatusType
{
	IDLE,					// DO NOTHING.
	NORMAL,					// WORKING WELL.
	DONE,					// WORK DONE.
	STOPPING,				// STOPPING.
	STOPPED,				// STOPPED.
	//EXCEPTION,			// EXCEPTION OCCURED.
	EXCEPTION_TERMINATED,	// TERMINATED DUE TO AN EXCEPTION.
};

enum ExceptionType {
	OCCURED
};

struct StatusMonitor {
	StatusType st_type;
	int progress; // 0 ~ 100 in %
	string exception;
	ExceptionType ex_type;

	StatusMonitor() {
		st_type = StatusType::IDLE;
		progress = 0;
		exception = "";
		ex_type = ExceptionType::OCCURED;
	}
};

namespace tata {

	enum class AnalysisPlan : unsigned int {
		CXR_LUNG,				// 0 (LUNG VOLUMETRY) 
		CXR_BONE,				// 1 (BONE)
		CXR_HEART,				// 2 (HEART VOLUMETRY)
		CXR_AORTA,				// 3 (AORTA) 
		CXR_VASCULAR,			// 4 (Vessel) 
		CXR_COVID,				// 5 (COVID)
		CXR_AIRWAY,				// 6 (Airway)
		CXR_TB,					// 7 (Tuberculosis - TB)
		CXR_NTM					// 8 (Non-Tuberculosis Mycobacterium - NTM)
	};

	struct Point {
		short x, y;
	};

	struct Line {
		Point start;		// start is the x,y coordinate of the left most pixel of the line
		Point end;			// end is the x,y coordinate of the right most pixel of the line
	};

	struct PatientInfo {
		short age;
		string sex;
		PatientInfo() {
			age = 0;
			sex = "Unknown";
		}
	};

	struct PatientData {
		int img_size_x;
		int img_size_y;
		int img_size_z;

		float img_spacing_x;
		float img_spacing_y;
		float img_spacing_z;

		vector<short> img_original;
		
		PatientData() {
			img_size_x = 0;
			img_size_y = 0;
			img_size_z = 0;

			img_spacing_x = 0.0f;
			img_spacing_y = 0.0f;
			img_spacing_z = 0.0f;
		}
	};

	struct Layer {
		vector<short> v_tisepx_image;		// TiSepX image (HU value) inf the form of depth, width, height
		Point p_contour_info;				// contour information (x,y) of contour points in the image
		short area;							// area of TiSepX output image (ex: heart)
		short volume;						// volume of TiSepX output image (ex: heart)
		Line l_info;						// line information (start, end) of the line for aneurysm detection graph
		Line l_aneurysm;					// aneurysm line information (start, end) 
		vector<uint8_t> heatmap;			// heatmap information for the output image (ex: TB heatmap)
		vector<short> vascular_area_change; // a range of 11 numbers from 0 to 100  
		short l1_osteoprosis;				// L1 osteoprosis value
		Layer() {
			v_tisepx_image = { 0, 0, 0 };
			p_contour_info = { 0, 0 };
			area = 0;
			volume = 0;
			l_info = { {0,0}, {0,0} };
			l_aneurysm = { {0,0}, {0,0} };
			heatmap = {};
			vascular_area_change = {};
			l1_osteoprosis = 0;
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
		AIModel preprocessing;		// in case there is a preprocessing by model
		AIModel lung_segmentation;
		AIModel heart_segmentation;
		AIModel lung_regression;
		AIModel vessel_segmentation;
		AIModel bone_segmentation;
		AIModel TB;
		AIModel NTM;
		AIModel COVID;
		AIModel AORTA;
		AIModel EMPHYSEMA;
	};

	class TATA_API TataBase
	{
	public:
		
		static void clearCache();
		// constructors 
		TataBase() {};
		TataBase(const Configuration& config);
		// destructor
		~TataBase();

		bool tiSepStop();			//FIXME: Change the nam of the method
		StatusMonitor getStatus() {
			lock_guard<mutex> guard(_checkMutex);
			return _check;
		}
		// setters
		
	protected:
		
		bool checkStopSetProgress(int start, int end, float progress);
		bool checkStop();
		void clearThread();

		Configuration				m_config;
		StatusMonitor				_check;
		mutex						_checkMutex;
		vector<thread>				_thread;
	};


	class TATA_API Analysis : virtual public TataBase {
		// this class includes TiSepX translation as well regression models

	public:
		Analysis() {};
		Analysis(const Configuration& config);
		~Analysis();

		bool doAnalysis();
		void threadDoAnalysis();

		// setters 
		void setPatientInfo(const PatientInfo p_info);
		void setPatientData(const PatientData p_data);
		void setAnalysisPlan(const vector<AnalysisPlan> v_analysis_plan);

		//getters for meddeling 
		PatientInfo getPatientInfo();
		PatientData getPatientData();
		vector<AnalysisPlan> getAnalysisPlan();
		Configuration getConfig();


		// get results
		//layer name (ex:heart)   Layer info (tisepx img, area, vol etc.)	
		//			   \			  /				 
		//				\            /		
		unordered_map<string, Layer> getOutputLayers();


	private:
		PatientInfo								m_p_info;
		PatientData								m_p_data;
		unordered_map<string, Layer>			m_m_output_layers;
		vector<AnalysisPlan>					m_v_analysis_plan;
	};

}// end namespace tata