/*
Author: Monib Sediqi @ Medical IP Inc.
Email: monib.korea@gmail.com | kh.monib@gmail.com
Date: 2023-02-01
All rights reserved.
*/
#pragma once
// include the necessary headers
// std

#ifndef _STDX_H
#define _STDX_H

#include <iostream>
#include<experimental/filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>

// torch
#include <torch/torch.h>


// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


// locals 
#include "dataset/ReadRawDataset.h"
#include "utilities/Postprocessing.h"
#include "utilities/ImgUtils.h"

// easylogging++
#include "easylogging++.h"

// https://github.com/pytorch/examples/tree/main/cpp
// easy logging docs = https://github.com/amrayn/easyloggingpp


#endif // !_STDX_H