# TATA  & SegCore C++ API
TATA & SeCore API are C++ libraries for tissue separation from x-rays and 3D segmentation and analysis of CT scans. The APIs are designed to be simple, intuitive, and easy to integrate into existing. It incorporates the best practices for software development including the use of the right data structure, searching algorithms, and more. The aim is to develop C++ APIs that can be integrated seamlessly with the current software/product of Medical IP Inc

# API Overview
![Screenshot](SegCore_API.png)

## Features
* AI X-ray tissue separation
* X-ray of bone, lung, pulmonary vessels, aorta, and airway separation
* 3D segmentation of medical images
* 3D analysis of medical images
* Full Precision (FP32) and Half Precision (FP16) support
* Read and Write to raw file support
* Torchscrip Model support
* Encryption and Decryption of model files
* Multi-GPU support
* Multi-Thread support 
* Deep Learning Multi-Model support 
* Interactive segmentation (coming soon)
* DL Model re-training support (coming soon)
* Promptable segmentation (coming soon)
* Text to medical image generation and analysis (coming soon)

## TiSepX Model Support 
* Bone model
* Heart model
* Lung model
* Pulmonary vessel model
* Aorta model
* Airway model
* COVID model
* Tuberculosis model

## Installation
### Dependencies
* [opencv](https://opencv.org/) >= 3.4
* [openssl](https://www.openssl.org/) >= 1.1.1
* [libtorch](https://pytorch.org/) == 1.7.1
* [dcmtk] (https://dicom.offis.de/dcmtk.php.en) = 3.6.7 

#### Dependency Installation Guide 
* [dcmtk] (https://brandres.medium.com/setup-dcmtk-with-cmake-for-c-and-visual-studio-2019-development-c5b3a40c9a54)


### Build
Download the source code and build the solution file using visual studio. The solution file is located in the root directory of the project named AI.sln. 

## Author
 
Monib Sediqi @ Research & Science Department, Medical IP Inc. (monib.sediqi@medicalip.com)

Date: 2023-02-01

[Medical IP Inc.](https://www.medicalip.com/)
