#pragma once
#include "TataBase.h"

class Segmentation {
public:
		Segmentation(const tata::Configuration& config);
		~Segmentation();
		//setters
		void setInputImage(vector<short> v_input_image);
		void setInputSizeX(short size_x);
		void setInputSizeY(short size_y);
		void setInputSizeZ(short size_z);

		void setImgSpacingX(float spacing_x);
		void setImgSpacingY(float spacing_y);
		void setImgSpacingZ(float spacing_z);

		 bool run(SegTask seg_task);

		 //getters
		 vector<float> getOutputImage();
		 short getOutputSizeX();
		 short getOutputSizeY();
		 short getOutputSizeZ();

		 float getImgSpacingX();
		 float getImgSpacingY();
		 float getImgSpacingZ();
		 float getArea();
		 //vector<int64_t> getOutputImageSize();
private:
	tata::Configuration			m_config;
	vector<short>				m_v_input_image;
	short						m_input_size_x;
	short						m_input_size_y;
	short						m_input_size_z;

	float						m_img_spacing_x;
	float						m_img_spacing_y;
	float						m_img_spacing_z;

	vector<float>				m_v_output_image;
	short 						m_output_size_x;
	short						m_output_size_y;
	short						m_output_size_z;

	float						m_area;

	//vector<int64_t>				m_output_image_size; // z,y,x



};