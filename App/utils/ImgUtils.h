/*
Author: Monib Sediqi @ Medical IP Inc.
Email: monib.sediqi@medicalip.com | kh.monib@gmail.com
Date: 2023-02-01
All rights reserved.
*/
#pragma once
#ifndef _IMAGE_UTILS_H
#define _IMAGE_UTILS_H

// System includes
#include<iostream>

// Third party includes
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>

namespace imgutils {
	cv::Mat readImage2D(const std::string& imagePath);
	void showImage2D(cv::Mat& image);
}
#endif // !_IMAGE_UTILS_H
