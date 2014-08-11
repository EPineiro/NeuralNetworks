#include <iostream>
#include <vector>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <armadillo>

#include "NeuralNetwork.h"
#include "util/MNISTDataLoader.h"

using namespace std;
using namespace arma;
using namespace cv;

void visualize_data(vector<pair<mat, mat> > &data);
void train_net(vector<pair<mat, mat> > &training_mat, vector<pair<mat, mat> > &validation_data);
void evaluate_net(vector<pair<mat, mat> > &test_mat);

string NET_FILE_NAME = "<local dir>/NeuralNets/resources/net.dat";
string MNIST_DIR = "<local dir>/NeuralNets/resources/MNIST";
string MNIST_REDUCED_DIR = "<local dir>/NeuralNets/resources/mnist_reducido";
string MONITORED_DATA_DIR = "<local dir>/NeuralNets/resources/monitored_data";

int INPUT_LAYER_SIZE = 784;
int HIDDEN_LAYER_SIZE = 30;
int OUTPUT_LAYER_SIZE = 10;

int CANT_VALIDATION_ELEMENTS = 10000;
int CANT_TEST_ELEM = 10000;
//int CANT_TEST_ELEM = 2115;

int main(int argc, char** argv) {

	vector<pair<mat, mat> > training_data, validation_data, test_data;
	MNISTDataLoader::load_mnist_data(training_data, validation_data, test_data, MNIST_DIR, CANT_VALIDATION_ELEMENTS);

	//visualize_data(test_data);

	train_net(training_data, validation_data);
	evaluate_net(test_data);

	return 0;
}

void train_net(vector<pair<mat, mat> > &training_data, vector<pair<mat, mat> > &validation_data) {

	int sizes[] = { INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE };
	NeuralNetwork net(sizes, sizeof(sizes) / sizeof(int));

	net.set_monitor_training_cost(true);
	net.set_monitor_training_accuracy(true);
	net.set_monitor_validation_cost(true);
	net.set_monitor_validation_accuracy(true);

	net.stochastic_gradient_descent_training(training_data, 30, 10, 0.001, 1.0, 0.1, validation_data);

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

void visualize_data(vector<pair<mat, mat> > &data) {

	for(int i = 0; i < 10; i++) {

		uword index;
		data[i].second.max(index);

		cv::Mat image(28, 28, CV_32FC1);

		int k = 0;
		for(int r = 0; r < 28; r++) {
			for(int c = 0; c < 28; c++) {

				image.at<float>(r, c) = data[i].first(k++, 0);
			}
		}

		stringstream  ss;
		ss<<index<<"["<<i+1<<"]";
		namedWindow(ss.str());
		imshow(ss.str(), image);
	}

	waitKey(0);
}

