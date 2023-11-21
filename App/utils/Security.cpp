/*
Author: Monib Sediqi @ Medical IP Inc.
Email: monib.sediqi@medicalip.com | kh.monib@gmail.com
Date: 2023-02-01
All rights reserved.
*/

#include "Security.h"

/////////////////////////////// Global Functions /////////////////////////////////////////
void encrypt_file(const string& in_file, const string& out_file, const string key) {
	// Read input file
	ifstream input(in_file, ios::binary);
	std::stringstream buffer;
	buffer << input.rdbuf();
	std::string input_str = buffer.str();
	input.close();

	// Generate a random IV initialization vector
	unsigned char iv[EVP_MAX_IV_LENGTH];
	RAND_bytes(iv, EVP_MAX_IV_LENGTH);

	// Initializ the encryption context
	EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
	EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, (unsigned char*)key.c_str(), iv);

	// Encrypt the input string
	int out_len;
	unsigned char* out_buf = (unsigned char*)malloc(input_str.size() + EVP_MAX_BLOCK_LENGTH);
	EVP_EncryptUpdate(ctx, out_buf, &out_len, (unsigned char*)input_str.c_str(), input_str.size());
	int final_len;
	EVP_EncryptFinal_ex(ctx, out_buf + out_len, &final_len);
	out_len += final_len;

	// Write the encrypted data to the output file
	std::ofstream output(out_file, std::ios::binary);
	output.write((char*)iv, EVP_MAX_IV_LENGTH);
	output.write((char*)out_buf, out_len);
	output.close();

	// clean up
	EVP_CIPHER_CTX_free(ctx);
	free(out_buf);
}
void decrypt_file(const string& in_file, const string& out_file, const string key) {
	// read the input file
	ifstream input(in_file, ios::binary);
	if (!input.is_open()) {
		throw runtime_error("Could not open input file");
	};
	// read the IV from the input file
	unsigned char iv[EVP_MAX_IV_LENGTH];
	input.read((char*)iv, EVP_MAX_IV_LENGTH);

	// Initialize the decryption context
	EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
	EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, (unsigned char*)key.c_str(), iv);

	// Read the encrypted data from the input file
	std::stringstream buffer;
	buffer << input.rdbuf();
	std::string input_str = buffer.str();
	input.close();

	// Decrypt the input string
	int out_len;
	unsigned char* out_buf = (unsigned char*)malloc(input_str.size() + EVP_MAX_BLOCK_LENGTH);
	EVP_DecryptUpdate(ctx, out_buf, &out_len, (unsigned char*)input_str.c_str(), input_str.size());
	int final_len;
	EVP_DecryptFinal_ex(ctx, out_buf + out_len, &final_len);
	out_len += final_len;


	// Write the decrypted data to the output file
	std::ofstream output(out_file, std::ios::binary);
	if (!output.is_open()) {
		throw std::runtime_error("Failed to open the output file.");
	}
	output.write((char*)out_buf, out_len);
	output.close();

	// Clean up
	EVP_CIPHER_CTX_free(ctx);
	free(out_buf);
}