#include <iostream>
#include <vector>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <armadillo>

#include "NeuralNetwork.h"
#include "util/MNISTDataLoader.h"
#include "util/images/ImageProcessor.h"
#include "util/images/ImageLoader.h"

using namespace std;
using namespace arma;
using namespace cv;

void visualize_data(vector<pair<mat, mat> > &data, int size, float factor = 1);
void train_net(vector<pair<mat, mat> > &training_mat, vector<pair<mat, mat> > &validation_data);
void evaluate_net(vector<pair<mat, mat> > &test_mat);

void test_video();

string NET_FILE_NAME = "resources/net.dat";
string MNIST_DIR = "resources/MNIST";
string MNIST_REDUCED_DIR = "resources/mnist_reducido";
string MONITORED_DATA_DIR = "resources/monitored_data";
string SIGN_IMAGE_DIR ="resources/signs_images";

int INPUT_LAYER_SIZE = 2500;
int HIDDEN_LAYER_SIZE = 100;
int OUTPUT_LAYER_SIZE = 5;

int CANT_VALIDATION_ELEMENTS = 10000;
int CANT_TEST_ELEM = 10000;
//int CANT_TEST_ELEM = 2115;

int main(int argc, char** argv) {

	ImageLoader loader;
	ImageProcessor processor;
	vector<pair<mat, mat> > training_data, validation_data, test_data;
	//MNISTDataLoader::load_mnist_data(training_data, validation_data, test_data,	MNIST_DIR, CANT_VALIDATION_ELEMENTS);
	loader.load_image_data(training_data, validation_data, test_data, SIGN_IMAGE_DIR, CANT_VALIDATION_ELEMENTS);

	//visualize_data(training_data, 50, 255);

	train_net(training_data, validation_data);
	//evaluate_net(test_data);

	//test_video();

	//processor.process_image_dir(SIGN_IMAGE_DIR);

	return 0;
}

void train_net(vector<pair<mat, mat> > &training_data, vector<pair<mat, mat> > &validation_data) {

	int sizes[] = { INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE };
	NeuralNetwork net(sizes, sizeof(sizes) / sizeof(int));

	net.set_monitor_training_cost(true);
	net.set_monitor_training_accuracy(true);
	net.set_monitor_validation_cost(false);
	net.set_monitor_validation_accuracy(false);

	TrainingParams params;
	params.cant_epochs = 50;
	params.mini_batch_size = 10;
	params.eta = 0.1;
	params.lambda = 1.0;
	params.mu = 0.1;

	net.stochastic_gradient_descent_training(training_data, params, validation_data);

	net.print();
	net.save(NET_FILE_NAME);
	net.save_monitored_data(MONITORED_DATA_DIR);
}

void evaluate_net(vector<pair<mat, mat> > &test_data) {

	NeuralNetwork net;
	net.load(NET_FILE_NAME);
	//net.print();

	double corrects = net.evaluate(test_data);
	double percentage = (corrects / CANT_TEST_ELEM) * 100;
	cout << "corrects: " << corrects << ", percentage: " << percentage << "%" << endl;

	cout<<"total cost: "<<net.calcule_total_cost(test_data, 0);
}

void visualize_data(vector<pair<mat, mat> > &data, int size, float factor) {

	cout<<"visualizing data"<<endl;
	random_shuffle(data.begin(), data.end());

	for(int i = 0; i < 10; i++) {

		uword index;
		data[i].second.max(index);

		cv::Mat image(size, size, 0);

		int k = 0;
		for(int r = 0; r < size; r++) {
			for(int c = 0; c < size; c++) {

				image.at<uchar>(r, c) = data[i].first(k++, 0) * factor;
			}
		}

		stringstream  ss;
		ss<<index<<"["<<i+1<<"]";
		namedWindow(ss.str());
		imshow(ss.str(), image);
	}

	waitKey(0);
}

void test_video() {

	VideoCapture cap(CV_CAP_ANY);
	ImageProcessor processor;
	ImageLoader loader;
	NeuralNetwork net;
	net.load(NET_FILE_NAME);

	//net.visualize_hidden_units(1, 50);

	if (!cap.isOpened()) {
		cout << "Failed to initialize camera\n";
		return;
	}

	namedWindow("CameraCapture");
	namedWindow("ProcessedCapture");

	cv::Mat frame;
	while (true) {

		cap >> frame;

		cv::Mat processedFrame = processor.process_image(frame);

		if(processedFrame.rows * processedFrame.cols == INPUT_LAYER_SIZE) {

			mat input = loader.to_arma_mat(processedFrame);

			int label = net.predict(input);

			if(label == 0)
				putText(frame, "A", Point(500, 300), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(0), 3, 8);
			else if(label == 1)
				putText(frame, "E", Point(500, 300), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(0), 3, 8);
			else if(label == 2)
				putText(frame, "I", Point(500, 300), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(0), 3, 8);
			else if(label == 3)
				putText(frame, "O", Point(500, 300), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(0), 3, 8);
			else if(label == 4)
				putText(frame, "U", Point(500, 300), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(0), 3, 8);
		}

		imshow("CameraCapture", frame);
		imshow("ProcessedCapture", processedFrame);

		int key = waitKey(5);

		if(key == 13) {
			imwrite("captura.jpg", frame);
		}
		if (key == 27)
			break;
	}

	destroyAllWindows();
}
