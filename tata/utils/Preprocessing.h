/*
Author: Monib Sediqi @ Medical IP Inc.
Email: monib.korea@gmail.com | kh.monib@gmail.com
Date: 2023-02-01
All rights reserved.
*/
#ifndef _PREPROCESSING_H
#define _PREPROCESSING_H

//System includes
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

//Third party includes
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/util/Exception.h>

using namespace std;
using namespace torch;

/////////////////////////////////////// Global Functions ///////////////////////////////////
enum class NormMethod : unsigned int {
	MINMAX,			// 0
	ZSCORE,			// 1
	PERCENTILE,		// 2
	NONE,			// 3
};

struct NormParams {
	double_t min, max, avg, std;
};
struct Result {
	vector<short> v_pixel_data;
	float min, max;
};


// ----------------------------- VECTOR OPERATIONS ------------------------------------
float_t vecAvg(vector<float_t>& v);
float_t vecStd(vector<float_t>& v);

std::vector<short> readRawFile1D(std::string& raw_file_path, int d, int w, int h);

// ----------------------------- TENSOR OPERATIONS ------------------------------------
float calculatePercentile(torch::Tensor& tensor, float percentile);
Tensor normalizeTorch(torch::Tensor& t_data, NormMethod norm_method);
Tensor resizeKeepRatioXray(Tensor& t_img, const int target_size);
Tensor padImageXray(torch::Tensor& t_img, const int target_size, const int pad_value = -1024);


#endif // !_PREPROCESSING_H

