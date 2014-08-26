/*
 * ImageLoader.cpp
 *
 *  Created on: Aug 13, 2014
 *      Author: epineiro
 */

#include "ImageLoader.h"

using namespace std;
using namespace arma;
using namespace cv;

ImageLoader::ImageLoader() {
	// TODO Auto-generated constructor stub

}

ImageLoader::~ImageLoader() {
	// TODO Auto-generated destructor stub
}

void ImageLoader::load_image_data(vector<pair<mat, mat> > &training_mat,
		vector<pair<mat, mat> > &validation_mat,
		vector<pair<mat, mat> > &test_mat, const string &data_dir,
		int cant_validation_elems) {

	cout<<"loading training images"<<endl;
	string training_dir(data_dir);
	training_dir.append("/training");

	DIR *dp;
	struct dirent *dirp;

	if ((dp = opendir(training_dir.c_str())) == NULL) {
		cout << "Error opening " << training_dir << endl;
	}

	while ((dirp = readdir(dp)) != NULL) {

		string fileName = string(dirp->d_name);

		if (fileName.find_first_of('.') > 0) {

			stringstream imageDir;
			imageDir << training_dir << "/" << fileName;

			cv::Mat image = imread(imageDir.str(), CV_LOAD_IMAGE_GRAYSCALE);

			mat data = to_arma_mat(image);

			mat label = vectorize_label(fileName.substr(0, 1));

			training_mat.push_back(make_pair(data, label));
		}
	}

	cout<<"training images loaded"<<endl;
}

mat ImageLoader::vectorize_label(const string &sign_name) {

	mat result = zeros(number_of_classes, 1);

	if(sign_name == "A")
		result(0) = 1;
	else if(sign_name == "E")
		result(1) = 1;
	else if(sign_name == "I")
		result(2) = 1;
	else if(sign_name == "O")
		result(3) = 1;
	else if(sign_name == "U")
		result(4) = 1;

	return result;
}

mat ImageLoader::to_arma_mat(const cv::Mat &opencv_mat) {

	mat data(opencv_mat.rows * opencv_mat.cols, 1);

	int k = 0;
	for (int i = 0; i < opencv_mat.rows; i++) {
		for (int j = 0; j < opencv_mat.cols; j++) {

			data(k++, 0) = ((float) opencv_mat.at<uchar> (i, j)) / 255;
		}
	}

	return data;
}
