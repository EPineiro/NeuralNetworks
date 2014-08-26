/*
 * ImageProcessor.h
 *
 *  Created on: Aug 12, 2014
 *      Author: epineiro
 */

#ifndef IMAGEPROCESSOR_H_
#define IMAGEPROCESSOR_H_

#include <dirent.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <armadillo>

class ImageProcessor {
public:
	ImageProcessor();
	virtual ~ImageProcessor();

	void process_image_dir(const std::string &dir);
	cv::Mat process_image(const cv::Mat &image);

	bool crop_image_around_face(const cv::Mat &image, cv::Mat &output);
	cv::Mat filter_skin_tone(const cv::Mat &image);
	cv::Mat get_closed_image(const cv::Mat &image);
	cv::Mat mask_image(const cv::Mat &image, const cv::Mat &mask);
	cv::Mat decrease_resolution(const cv::Mat &image);
	cv::Mat zca_whitening(const cv::Mat &image);
};

#endif /* IMAGEPROCESSOR_H_ */
