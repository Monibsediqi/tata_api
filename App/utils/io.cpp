#include "io.h"

#define PRINT(x) std::cout<< x <<std::endl;
#define INFO(x) std::cout<<"INFO: " <<x<<std::endl;
#define ERR(x) std::cerr << "ERROR: " << x << std::endl;

///////////////////////////// input (read) part /////////////////////////////////
vector<short> readRawFile1D(string& raw_file_path, int d, int w, int h) {
	std::ifstream input_file(raw_file_path, std::ios::in | std::ios::binary);

	std::vector<short> v_pixel_val;
	if (!input_file) {
		ERR("Error: Failed to open file. " << raw_file_path);
		ERR("Returning an empty vector. ");
		return v_pixel_val; //return empty vector (nothing)
	};

	if (input_file.is_open()) {
		PRINT("Openning File successful. ");
		int16_t data; // size of the dicom data is 16 bit (2bytes)

		int count = 0;
		int img_resolution = d * w * h;
		while (input_file.read(/*buffer*/(char*)&data, /*buffer size*/sizeof(data))) {

			v_pixel_val.push_back((short)(data));
			++count;
			if (count > img_resolution) {
				throw std::invalid_argument(std::string("Wrong resolution value. Image size is larger than ") +
					std::to_string(d) + " x " + std::to_string(w) + "x" + std::to_string(h));
			}
		};
		input_file.close();
	}
	PRINT("Reading file successful.");
	return v_pixel_val;
};

///////////////////////////// output (write) part ///////////////////////////////
void writeRawData1D(vector<int16_t> data, string filename) {

	INFO("checking 200 pixels value: ");
	int counter = 0;
	for (auto x : data) {
		std::cout << x << " ";
		counter++;
		if (counter == 50) {
			break;
		}
	}
	auto file = std::ofstream(filename, std::ios::out | std::ios::binary);
	if (file.is_open()) {
		file.write((char*)&data[0], sizeof(int16_t) * data.size());
		file.close();
	}
}
void writeRawData3D(vector<vector<vector<int16_t>>> v_data, string filename) {
	auto outfile = ofstream(filename, ios::out | ios::binary);
	for (int i = 0; i < v_data.size(); i++) {
		for (int j = 0; j < v_data[0].size(); j++) {
			for (int k = 0; k < v_data[0][0].size(); k++) {
				short value = static_cast<short>(v_data[i][j][k]);
				outfile.write((char*)&value, sizeof(short));
			}
		}
	}
	outfile.close();
	INFO("file sved to: " << filename);
}