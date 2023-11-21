/*
Author: Monib Sediqi @ Medical IP Inc.
Email: monib.sediqi@medicalip.com | kh.monib@gmail.com
Date: 2023-02-01
All rights reserved.
*/

#ifndef _SECURITY_H
#define _SECURITY_H

// System includes
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>
#include<experimental/filesystem>

// third party includes 
//openssl
#include <openssl/evp.h>
#include <openssl/rand.h>


using namespace std;

void encrypt_file(const string& in_file, const string& out_file, const string key);
void decrypt_file(const string& in_file, const string& out_file, const string key);

#endif // !_SECURITY_H