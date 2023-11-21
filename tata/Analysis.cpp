#include "TataBase.h"
#include "Segmentation.h"
#include "Regression.h"
#include "utils/Preprocessing.h"
#include "utils/Postprocessing.h"
#include "utils/Optimization.h"


//third party includes
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>


using namespace std;
using namespace torch;
using namespace tata;

#define PRINT(x) cout <<"PRINT: " << x << endl;
#define INFO(x) cout << "INFO: " << x << endl;
#define ERR(x) cerr << "ERROR: " << x << endl;



Analysis::Analysis(const Configuration& config) : TataBase(config) {
	clearCache();
	clearThread();
}
Analysis::~Analysis() {

};

bool Analysis::doAnalysis() {
	Segmentation seg_module(m_config);
	seg_module.setInputImage(m_p_data.img_original);
	seg_module.setInputSizeX(m_p_data.img_size_x);
	seg_module.setInputSizeY(m_p_data.img_size_y);
	seg_module.setInputSizeZ(m_p_data.img_size_z);

	seg_module.setImgSpacingX(m_p_data.img_spacing_x);
	seg_module.setImgSpacingY(m_p_data.img_spacing_y);
	seg_module.setImgSpacingZ(m_p_data.img_spacing_z);

	Regression reg_module(m_config);
	reg_module.setInputImgSizeX(m_p_data.img_size_x);
	reg_module.setInputImgSizeY(m_p_data.img_size_y);
	reg_module.setInputImgSizeZ(m_p_data.img_size_z);

	bool seg_successful;
	bool reg_successful;

	for (auto& a_plan : m_v_analysis_plan) {
		Layer output_layer;
		if (a_plan == AnalysisPlan::CXR_LUNG) {
			seg_successful = seg_module.run(SegTask::LUNG_SEGMENTATION);
			if (seg_successful) {

				vector<float> v_output = seg_module.getOutputImage();
				output_layer.v_tisepx_image = vector<short>(v_output.begin(), v_output.end());
				output_layer.area = seg_module.getArea();
				INFO("LUNG_SEGMENTATION successful.")

				//regression module for lung volumetry
				reg_module.setInputImgSizeX(seg_module.getOutputSizeX());
				reg_module.setInputImgSizeY(seg_module.getOutputSizeY());
				reg_module.setInputImgSizeZ(seg_module.getOutputSizeZ());
				reg_module.setInputImage(seg_module.getOutputImage());

				reg_successful = reg_module.run(RegTask::LUNG_VOLUME);
				if (reg_successful) {
					output_layer.volume = reg_module.getVolume();

					auto it = m_m_output_layers.find("lung");
					if (it != m_m_output_layers.end()) {
						// key exist
						it->second = output_layer;
					}
					else {
						// key does not exit
						m_m_output_layers.insert({ "lung", output_layer });
					}
					m_m_output_layers.insert(make_pair("lung", output_layer));
					INFO("LUNG_VOLUME successful.")
				}
				else {
					ERR("Lung Volume Regression failed");
				}
			}
			else {
				ERR("Lung Segmentation failed");
			}
		}
		else if (a_plan == AnalysisPlan::CXR_BONE) {
			seg_successful = seg_module.run(SegTask::BONE_SEGMENTATION);
			if (seg_successful) {
				output_layer.area = seg_module.getArea();
				vector<float> v_output = seg_module.getOutputImage();
				output_layer.v_tisepx_image = vector<short>(v_output.begin(), v_output.end());
				m_m_output_layers.insert(make_pair("bone", output_layer));
				INFO("Completed CXR_BONE Analysis.")
			}
			else {
				ERR("CXR_BONE Segmentation failed")
			}
		}
		else if (a_plan == AnalysisPlan::CXR_HEART) {
			seg_successful = seg_module.run(SegTask::HEART_SEGMENTATION);
			if (seg_successful) {
				output_layer.area = seg_module.getArea();
				vector<float> v_output = seg_module.getOutputImage();
				output_layer.v_tisepx_image = vector<short>(v_output.begin(), v_output.end());
				m_m_output_layers.insert(make_pair("heart", output_layer));
				INFO("Completed CXR_HEART Analysis.")
			}
			else {
				ERR("CXR_HEART Segmentation failed")
			}
		}
		else if (a_plan == AnalysisPlan::CXR_AORTA) {
			seg_successful = seg_module.run(SegTask::AORTA_SEGMENTATION);
			if (seg_successful) {
				output_layer.area = seg_module.getArea();
				vector<float> v_output = seg_module.getOutputImage();
				output_layer.v_tisepx_image = vector<short>(v_output.begin(), v_output.end());
				m_m_output_layers.insert(make_pair("aorta", output_layer));
				INFO("Completed CXR_AORTA Analysis.")
			}
			else {
				ERR("CXR_AORTA Segmentation failed")
			}
		}
		else if (a_plan == AnalysisPlan::CXR_VASCULAR) {
			PRINT("Performing both lung segmentation and vascular segmentation.")
				seg_successful = seg_module.run(SegTask::LUNG_SEGMENTATION);
			if (seg_successful) {
				output_layer.area = seg_module.getArea();
				vector<float> v_output = seg_module.getOutputImage();
				output_layer.v_tisepx_image = vector<short>(v_output.begin(), v_output.end());

				// check whether lung sementation has already been performed or not. 
				auto it = m_m_output_layers.find("lung");
				if (it != m_m_output_layers.end()) {
					// key exist
					it->second = output_layer;
				}
				else {
					// key does not exit
					m_m_output_layers.insert({ "lung", output_layer });
				}
				INFO("CXR_VASCULAR: Completed LUNG_SEGMENTATION Analysis.");

				seg_module.setInputImage(output_layer.v_tisepx_image);
				seg_module.setInputSizeX(seg_module.getOutputSizeX());
				seg_module.setInputSizeY(seg_module.getOutputSizeY());
				seg_module.setInputSizeZ(seg_module.getOutputSizeZ());

				PRINT("set input img!")
				seg_successful = seg_module.run(SegTask::VASCULAR_SEGMENTATION);
				if (seg_successful) {
					output_layer.area = seg_module.getArea();
					vector<float> v_output = seg_module.getOutputImage();
					output_layer.v_tisepx_image = vector<short>(v_output.begin(), v_output.end());
					m_m_output_layers.insert(make_pair("vascular", output_layer));
					INFO("Completed CXR_VASCULAR Analysis.");

					// reset the segmentation module -> change it to a method
					seg_module.setInputImage(m_p_data.img_original);
					seg_module.setInputSizeX(m_p_data.img_size_x);
					seg_module.setInputSizeY(m_p_data.img_size_y);
					seg_module.setInputSizeZ(m_p_data.img_size_z);
				}
				else {
					ERR("CXR_VASCULAR Segmentation failed")
				}
			}
		}
		else if (a_plan == AnalysisPlan::CXR_AIRWAY) {
			seg_successful = seg_module.run(SegTask::AIRWAY_SEGMENTATION);
			if (seg_successful) {
				output_layer.area = seg_module.getArea();
				vector<float> v_output = seg_module.getOutputImage();
				output_layer.v_tisepx_image = vector<short>(v_output.begin(), v_output.end());
				m_m_output_layers.insert(make_pair("airway", output_layer));
				INFO("Completed CXR_VASCULAR Analysis.")
			}
			else {
				ERR("CXR_AIRWAY Segmentation failed")
			}
		}
		else if (a_plan == AnalysisPlan::CXR_TB) {
			seg_successful = seg_module.run(SegTask::TB_SEGMENTATION);
			if (seg_successful) {
				output_layer.area = seg_module.getArea();
				vector<float> v_output = seg_module.getOutputImage();
				output_layer.v_tisepx_image = vector<short>(v_output.begin(), v_output.end());
				m_m_output_layers.insert(make_pair("tb", output_layer));
				INFO("Completed CXR_TB Analysis.")
			}
			else {
				ERR("CXR_TB Segmentation failed")
			}
		}
		else if (a_plan == AnalysisPlan::CXR_NTM) {
			seg_successful = seg_module.run(SegTask::NTM_SEGMENTATION);
			if (seg_successful) {
				output_layer.area = seg_module.getArea();
				vector<float> v_output = seg_module.getOutputImage();
				output_layer.v_tisepx_image = vector<short>(v_output.begin(), v_output.end());
				m_m_output_layers.insert(make_pair("ntm", output_layer));
				INFO("Completed CXR_NTM Analysis.")
			}
			else {
				ERR("CXR_NTM Segmentation failed")
			}
		}
		else if (a_plan == AnalysisPlan::CXR_COVID) {
			PRINT("Performing both lung segmentation and Covid segmentation.")
				seg_successful = seg_module.run(SegTask::LUNG_SEGMENTATION);
			if (seg_successful) {
				output_layer.area = seg_module.getArea();
				vector<float> v_output = seg_module.getOutputImage();
				output_layer.v_tisepx_image = vector<short>(v_output.begin(), v_output.end());

				// check whether lung sementation has already been performed or not. 
				auto it = m_m_output_layers.find("lung");
				if (it != m_m_output_layers.end()) {
					// key exist
					it->second = output_layer;
				}
				else {
					// key does not exit
					m_m_output_layers.insert({ "lung", output_layer });
				}
				INFO("CXR_VASCULAR: Completed LUNG_SEGMENTATION Analysis.");

				seg_module.setInputImage(output_layer.v_tisepx_image);
				seg_module.setInputSizeX(seg_module.getOutputSizeX());
				seg_module.setInputSizeY(seg_module.getOutputSizeY());
				seg_module.setInputSizeZ(seg_module.getOutputSizeZ());

				PRINT("set input img!")
					seg_successful = seg_module.run(SegTask::COVID_SEGMENTATION);
				if (seg_successful) {
					output_layer.area = seg_module.getArea();
					vector<float> v_output = seg_module.getOutputImage();
					output_layer.v_tisepx_image = vector<short>(v_output.begin(), v_output.end());
					m_m_output_layers.insert(make_pair("covid", output_layer));
					INFO("Completed CXR_VASCULAR Analysis.");

					// reset the segmentation module -> change it to a method
					seg_module.setInputImage(m_p_data.img_original);
					seg_module.setInputSizeX(m_p_data.img_size_x);
					seg_module.setInputSizeY(m_p_data.img_size_y);
					seg_module.setInputSizeZ(m_p_data.img_size_z);
				}
				else {
					ERR("CXR_VASCULAR Segmentation failed")
				}
			}
		}
		else {
				PRINT("No analysis plan found");
				return false;
		}
	}
}
void Analysis::threadDoAnalysis(){};

// setters 
void Analysis::setPatientInfo(PatientInfo p_info) {
	m_p_info = p_info;
};
void Analysis::setPatientData(PatientData p_data) {
	m_p_data = p_data;
};
void Analysis::setAnalysisPlan(const vector<AnalysisPlan> v_analysis_plan) {
	m_v_analysis_plan = v_analysis_plan;
};

//getters for meddeling 
PatientInfo Analysis::getPatientInfo(){ 
	return m_p_info;
}
PatientData Analysis::getPatientData(){ 
	return m_p_data;
}
vector<AnalysisPlan> Analysis::getAnalysisPlan() {
	return m_v_analysis_plan;
};
Configuration Analysis::getConfig() {
	return m_config;
};

unordered_map<string, Layer> Analysis::getOutputLayers() {
	return m_m_output_layers;
}