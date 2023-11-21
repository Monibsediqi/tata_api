/*
Author: Monib Sediqi @ Medical IP Inc.
Email: monib.sediqi@medicalip.com | kh.monib@gmail.com
Date: 2023-02-01
All rights reserved.
*/

#include "ImgUtils.h"

namespace imgutils {
	cv::Mat readImage2D(const std::string& imagePath) {
		cv::Mat image = imread(imagePath, cv::IMREAD_COLOR);
		return image;
	}

	void showImage2D(cv::Mat& image) {

		cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
		cv::imshow("Display window", image);
		printf("Waiting for key stroke to exit.");
		cv::waitKey(0); // wait for a keystrok
	}
}