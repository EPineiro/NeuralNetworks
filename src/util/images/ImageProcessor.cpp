/*
 * ImageProcessor.cpp
 *
 *  Created on: Aug 12, 2014
 *      Author: epineiro
 */

#include "ImageProcessor.h"

using namespace std;
using namespace cv;

ImageProcessor::ImageProcessor() {
	// TODO Auto-generated constructor stub

}

ImageProcessor::~ImageProcessor() {
	// TODO Auto-generated destructor stub
}

void ImageProcessor::process_image_dir(const string &dir) {

	DIR *dp;
	struct dirent *dirp;

	stringstream ss_dir;
	ss_dir << dir << "/originals";

	if ((dp = opendir(ss_dir.str().c_str())) == NULL) {
		cout << "Error opening " << dir << endl;
	}

	while ((dirp = readdir(dp)) != NULL) {

		string fileName = string(dirp->d_name);

		if (fileName.find_first_of('.') > 0) {

			cout << "processing image: " << fileName << endl;

			stringstream imageDir;
			imageDir << dir << "/originals/" << fileName;

			cv::Mat image = imread(imageDir.str());

			cv::Mat output = process_image(image);

			stringstream outputFile;
			outputFile << dir << "/processed/" << fileName;
			imwrite(outputFile.str(), output);
		}
	}
}

Mat ImageProcessor::process_image(const Mat &image) {

	Mat output = image.clone();
	bool detected = crop_image_around_face(image, output);

	if(detected) {

		Mat mask = output.clone();
		mask = filter_skin_tone(mask);
		mask = get_closed_image(mask);

		output = mask_image(output, mask);
		output = decrease_resolution(output);
		output = zca_whitening(output);
	}

	return output;
}

bool ImageProcessor::crop_image_around_face(const Mat &image, Mat &output) {

	CascadeClassifier *face_detector = new CascadeClassifier();
	face_detector->load("resources/otros/lbpcascade_frontalface.xml");
	vector<Rect> face_detections;

	Rect result;

	face_detector->detectMultiScale(image, face_detections);

	delete face_detector;

	if (!face_detections.empty()) {

		result = face_detections.at(0);

		//enlarge the size, so the hands also appears in the image
		result.x -= 150;
		result.height += 150;
		result.width += 200;

		//check that the roi doesn't surpass image bounds
		if(result.x < 0)
			result.x = 0;
		if(result.y + result.height > image.rows)
			result.height = image.rows - result.y;
		if(result.x + result.width > image.cols)
				result.width = image.cols - result.x;

		output = image(result);

		return true;
	}
	else {
		return false;
	}
}

Mat ImageProcessor::filter_skin_tone(const Mat &image) {

	Mat newImage = image.clone();
	GaussianBlur(image, newImage, Size(5,5), 1.5);
	cvtColor(newImage, newImage, CV_BGR2YCrCb);

	//turn the image to a binary one, leaving only the skin tone
	inRange(newImage, Scalar(0, 135, 80), Scalar(255, 200, 140), newImage);

	return newImage;
}

Mat ImageProcessor::get_closed_image(const Mat &image) {

	Mat newImage = image.clone();
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(11,11));
	morphologyEx(image, newImage, MORPH_CLOSE, kernel, Point(-1, -1), 1);
	return newImage;
}

Mat ImageProcessor::mask_image(const Mat &image, const Mat &mask) {

	Mat maskedImage = image.clone();
	cvtColor(maskedImage, maskedImage, CV_BGR2GRAY);
	bitwise_and(maskedImage, mask, maskedImage);

	return maskedImage;
}

Mat ImageProcessor::decrease_resolution(const Mat &image) {

	Mat newImage = image.clone();
	resize(image, newImage, Size(50, 50), 0.5, 0.5, CV_INTER_AREA);
	return newImage;
}

Mat ImageProcessor::zca_whitening(const Mat &image) {

	//for this withening we need to do PCA to the image.
	//most of the functions needed for this are in armadillo,
	//so we need to pass the opencv Mat to an armadillo mat and viceversa.
	arma::mat data = arma::zeros<arma::mat>(1, image.rows * image.cols);
	int k = 0;
	for(int i = 0; i < image.rows; i++) {
		for(int j = 0; j < image.cols; j++) {

			data(k++) = image.at<uchar>(i, j);
		}
	}

	//PCA
	arma::mat U, V, sigma, xZCAWhite;
	arma::vec S;

	sigma = (data * data.t()) / data.n_cols;
	arma::svd(U, S, V, sigma);

	//ZCA whitening
	arma::mat result = 1 / arma::sqrt(S.diag() + 0.1);
	xZCAWhite =  U * result.diag() * (U.t() * data);

	//convert again to opencv Mat
	Mat output(image.rows, image.cols, 0);
	k = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			output.at<uchar>(i, j) = xZCAWhite(k++) * 255;
		}
	}

	return output;
}
