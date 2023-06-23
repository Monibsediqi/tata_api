/*
Author: Monib Sediqi @ Medical IP Inc.
Email: monib.sediqi@medicalip.com | kh.monib@gmail.com
Date: 2023-02-01
All rights reserved.
*/

#pragma once
#ifndef _POSTPROCESSING_H
#define _POSTPROCESSING_H

// System includes
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

// Third party includes
#include <torch/torch.h>

using namespace std;
using namespace torch;

/// ///////////////////////// Data Structures ////////////////////////////////////

struct PostprocessingParams {
	int d, h, w;
};

///////////////////////////// Global Functions /////////////////////////////////
vector<int16_t> convertFloatToInt16(const vector<float_t>);
vector<float_t> convertTensorToVector(const Tensor& tensor);
vector<vector<vector<float_t>>> convertTensorTo3DVector(const Tensor& tensor);
vector<vector<vector<int16_t>>> convertTensorTo3DVectorInt16(const torch::Tensor& tensor);



#endif // !_POSTPROCESSING_H