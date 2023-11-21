#pragma once
#include "TataBase.h"

class Regression {
public:
	Regression(const tata::Configuration& config);
	~Regression();
	// setters 
	//void setInputImageSize(const vector<int64_t> v_input_image_size); // z,y,x
	void setInputImage(const vector<float>& v_input_image); // z,y,x
	void setInputImgSizeX(const short& size_x);
	void setInputImgSizeY(const short& size_y);
	void setInputImgSizeZ(const short& size_z);
	void setArea(const float& area);
	void setPatientAge(const short& age);
	void setPatientSex(const string& sex);

	//gettters
	//vector<int64_t> getInputImageSize();
	bool run( RegTask reg_task);
	short getInputImgSizeX();
	short getInputImgSizeY();
	short getInputImgSizeZ();
	float getArea();
	short getPatientAge();
	string getPatientSex();
	short getVolume();

private:
	tata::Configuration			m_config;
	short						m_vol;
	vector<float>				m_v_input_image;
	short						m_input_size_x;
	short						m_input_size_y;
	short						m_input_size_z;
	float						m_area;
	short						m_patient_age;
	string						m_patient_sex;
	//vector<int64_t>				m_input_image_size;
};