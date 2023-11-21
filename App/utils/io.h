/*
Author: Monib Sediqi @ Medical IP Inc.
Email: monib.sediqi@medicalip.com | kh.monib@gmail.com
Date: 2023-02-01
All rights reserved.
*/

#ifndef _IO_H
#define _IO_H

// System includes
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

// Third party includes

// Project includes

using namespace std;

///////////////////////////// input (read) part /////////////////////////////////
vector<short> readRawFile1D(string& raw_file_path, int d, int w, int h);

///////////////////////////// output (write) part ///////////////////////////////
void writeRawData1D(vector<int16_t> data, string filename);
void writeRawData3D(vector<vector<vector<int16_t>>> v_data, string filename);

#endif // !_IO_H