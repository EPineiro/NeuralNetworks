/*
 * ImageLoader.h
 *
 *  Created on: Aug 13, 2014
 *      Author: epineiro
 */

#ifndef IMAGELOADER_H_
#define IMAGELOADER_H_

#include <dirent.h>

#include <opencv2/opencv.hpp>
#include <armadillo>

class ImageLoader {
private:
	static const int number_of_classes = 5;

	arma::mat vectorize_label(const std::string &sign_name);
public:
	ImageLoader();
	virtual ~ImageLoader();

	void load_image_data(
			std::vector<std::pair<arma::mat, arma::mat> > &training_mat,
			std::vector<std::pair<arma::mat, arma::mat> > &validation_mat,
			std::vector<std::pair<arma::mat, arma::mat> > &test_mat,
			const std::string &data_dir, int cant_validation_elems);

	arma::mat to_arma_mat(const cv::Mat &opencv_mat);
};

#endif /* IMAGELOADER_H_ */
