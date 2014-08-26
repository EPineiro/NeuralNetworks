/*
 * NeuralNetwork.cpp
 *
 *  Created on: Jul 25, 2014
 *      Author: epineiro
 */

#include "NeuralNetwork.h"

using namespace std;
using namespace arma;
using namespace cv;

/**
 * Constructor
 * @param int vector with the sizes of each layer, for example {5, 20, 10}
 * @param number of layers
 */
NeuralNetwork::NeuralNetwork(int sizes[], int num_layers) {
	layer_sizes = sizes;
	this->num_layers = num_layers;

	//initialize weights and biases with random matrices with a gaussian distribution
	//we don't need biases for the first layer (the input)
	arma_rng::set_seed_random();
	for (int i = 1; i < num_layers; i++) {

		biases.push_back(randn(layer_sizes[i], 1));
		weights.push_back(randn(layer_sizes[i], layer_sizes[i - 1]) / sqrt(layer_sizes[i - 1]));

		momentum_weights.push_back(zeros<mat>(weights[i - 1].n_rows, weights[i - 1].n_cols));
		momentum_biases.push_back(zeros<mat>(biases[i - 1].n_rows, biases[i - 1].n_cols));
	}
}

/**************************************************************************************************************/
/**
 * Pass an input through the net and get an ouput
 * @param input, armadillo matrix (n x 1) representing an input, where n is equal to number of features.
 * @return armadillo matrix (c x 1) with the ouput, where c is equal to number of classification classes,
 * 			The max value in this vector is equal to the class predicted by the net.
 */
mat NeuralNetwork::feed_forward(const mat &input) {

	mat activation = input;

	//weights and biases should always have the same number of elements
	for(int i = 0; i < (num_layers-1); i++) {

		activation = sigmoid((weights[i] * activation) + biases[i]);
	}

	return activation;
}

/*************************************************************************************************************************/
/**
 * Train the net using the stochastic gradient descent algorithm.
 * @param stl vector containing stl pairs of two armadillo matrix, one with the example (nuber of features x 1)
 * 		  and the other with the label of it vectorized in the form (number of classes x 1)
 * 		  with a one in the position of the label and ceros otherwise.
 * @param number of epochs to train
 * @param size of the minibatch to use
 * @param learning rate
 * @param regularization parameter
 * @param momentum coefficient
 * @param stl vector containing stl pairs of two armadillo matrix with the same format as the training mat.
 * 		  This param is optional, but if present is used to print the results of the net between each epoch.
 */
void NeuralNetwork::stochastic_gradient_descent_training(
		vector<pair<mat, mat> > &training_data, TrainingParams params,
		vector<pair<mat, mat> > test_data) {

	for(int i = 0; i < params.cant_epochs; i++) {

		cout<<"Training epoch: "<<i+1<<endl;

		random_shuffle(training_data.begin(), training_data.end());

		//generate mini batches
		for(size_t j = 0; j < training_data.size(); j += params.mini_batch_size) {

			size_t endPos = j + params.mini_batch_size;
			if(endPos >= training_data.size())
				endPos = training_data.size();

			update_mini_batch(training_data.begin() + j, training_data.begin() + endPos, params, training_data.size());
		}


		if(!test_data.empty()) {
			cout<<"total corrects in epoch "<<i+1<<": "<<this->evaluate(test_data)<<endl;;
		}


		if(monitor_training_cost)
			training_costs.push_back(calcule_total_cost(training_data, params.lambda));
		if(monitor_training_accuracy)
			training_accuracies.push_back(evaluate(training_data));
		if(monitor_validation_cost)
			validation_costs.push_back(calcule_total_cost(test_data, params.lambda));
		if(monitor_validation_accuracy)
			validation_accuracies.push_back(evaluate(test_data));
	}
}

/***********************************************************************************************************************/
/**
 * Private method to updates the weights and biases of the net using the given mini batch of examples.
 * @param stl iterator of the training vector indicating where the mini batch begins in the data.
 * @param stl iterator of the training vector indicating where the mini bath ends in the data.
 * @param learning rate
 * @param regularization parameter
 * @param momentum coefficient
 * @param size of the mini batch
 * @param total number of training examples
 */
void NeuralNetwork::update_mini_batch(
		vector<pair<mat, mat> >::const_iterator begin,
		vector<pair<mat, mat> >::const_iterator end,
		TrainingParams params, int training_size) {

	vector<mat> nabla_w, nabla_b;

	//initialize matrices to accumulate gradients
	for (int i = 0; i < (num_layers - 1); i++) {

		nabla_w.push_back(zeros<mat>(weights[i].n_rows, weights[i].n_cols));
		nabla_b.push_back(zeros<mat>(biases[i].n_rows, biases[i].n_cols));
	}

	for(; begin != end; begin++) {

		//use back propagation to calcule gradients
		vector<mat> delta_nabla_w, delta_nabla_b;
		back_propagation(begin->first, begin->second, delta_nabla_w, delta_nabla_b);

		//acumulate partial gradients
		for (int i = 0; i < (num_layers - 1); i++) {

			nabla_w[i] += delta_nabla_w[i];
			nabla_b[i] += delta_nabla_b[i];
		}
	}

	//update weights and biases with gradients
	for (int i = 0; i < (num_layers - 1); i++) {

		//weights[i] = ((1 - (eta * (lambda/training_size))) * weights[i]) - ((eta / mini_batch_size) * nabla_w[i]);
		//biases[i] = biases[i] - ((eta / mini_batch_size) * nabla_b[i]);

		momentum_weights[i] = (params.mu * momentum_weights[i]) - ((params.eta / params.mini_batch_size) * nabla_w[i]);
		momentum_biases[i] = (params.mu * momentum_biases[i]) - ((params.eta / params.mini_batch_size) * nabla_b[i]);

		weights[i] = ((1 - (params.eta * (params.lambda / training_size))) * weights[i]) + momentum_weights[i];
		biases[i] = biases[i] + momentum_biases[i];
	}
}

/*********************************************************************************************************************************/
/**
 * Private method to calculate the gradients of the cost function with respect to the weights and biases using the backpropagation algorithm.
 * @param armadillo matrix (number of features x 1) representing an example.
 * @param armadillo matrix (number of classes x 1) representing the vectorized label.
 * @param stl vector of armadillo matrix used to store the gradients with respect to the weights.
 * 		  The size of each matrix is similar to the size of each weight matrix for each layer (the correspondence is one to one).
 *  @param stl vector of armadillo matrix used to store the gradients with respect to the biases.
 * 		  The size of each matrix is similar to the size of each biases vector for each layer (the correspondence is one to one).
 */
void NeuralNetwork::back_propagation(const mat &x, const mat &y, vector<mat> &delta_nabla_w, vector<mat> &delta_nabla_b) {

	vector<mat> activations, zs;
	zs.push_back(x);
	activations.push_back(x);

	//initialize accumulators and make feedforward pass
	for (int i = 0; i < (num_layers - 1); i++) {

		delta_nabla_w.push_back(zeros<mat>(weights[i].n_rows, weights[i].n_cols));
		delta_nabla_b.push_back(zeros<mat>(biases[i].n_rows, biases[i].n_cols));
		mat z = (weights[i] * activations[i]) + biases[i];
		zs.push_back(z);
		activations.push_back(sigmoid(z));
	}

	//backward pass
	//first we calcule cost for last layer
	mat delta = (activations.back() - y);

	//calcule deltas and partial gradients for remaining layers from back to front
	for(int i = (num_layers - 2); i >= 0; i--) {

		delta_nabla_b[i] = delta;
		delta_nabla_w[i] = delta * activations[i].t();
		delta = (weights[i].t() * delta) % sigmoid_prime(zs[i]);
	}

}
/**************************************************************************************************************************************/
/**
 * Calcule total cost of prediction for the examples given.
 * @param stl vector containing stl pairs of two armadillo matrix, one with the example (nuber of features x 1)
 * 		  and the other with the label of it vectorized in the form (number of classes x 1)
 * 		  with a one in the position of the label and ceros otherwise.
 * @param regularization parameter
 */
double NeuralNetwork::calcule_total_cost(vector<pair<mat, mat> > &data, double lambda) {

	double cost = 0.0;

	for(vector<pair<mat, mat> >::const_iterator it = data.begin(); it != data.end(); it++) {

		mat output = this->feed_forward(it->first);
		mat result = ((-it->second.t() * arma::log(output)) - ((1 - it->second).t() * arma::log(1 - output)));

		cost += result(0,0) / data.size();
	}

	//regularize cost
	double reg_factor = 0.0;
	for (int i = 0; i < (num_layers - 1); i++) {

		reg_factor += pow(arma::norm(weights[i]), 2);
	}

	cost += 0.5 * (lambda/data.size()) * reg_factor;

	return cost;
}

/**************************************************************************************************************************************/
/**
 * Evaluates the network predictions with respect to the given test data.
 * @param stl vector containing stl pairs of two armadillo matrix, one with the example (nuber of features x 1)
 * 		  and the other with the label of it vectorized in the form (number of classes x 1)
 * 		  with a one in the position of the label and ceros otherwise.
 * @return number of examples corrected predicted.
 */
int NeuralNetwork::evaluate(const vector<pair<mat, mat> > &test_data) {

	int total_correct_classifications = 0;

	for(vector<pair<mat, mat> >::const_iterator it = test_data.begin(); it != test_data.end(); it++) {

		//pass the example through the net and store the index in the vector of activations of the output layer with the max value.
		//the neural network's output is assumed to be the index of whichever neuron in the final layer has the highest activation.
		//the index is equal to the number we use as a label for that example.
		uword indexPrediction, indexLabel;
		it->second.max(indexLabel);

		indexPrediction = this->predict(it->first);

		//if the label is equal to the prediction, we score one hit as correct
		if(indexLabel == indexPrediction)
			total_correct_classifications++;
	}

	return total_correct_classifications;
}

/**************************************************************************************/
/**
 * Predict a label for a given input.
 * @param armadillo matrix (number of features x 1) representing the input
 * @return predicted label as index of the class in the output layer vector.
 */
int NeuralNetwork::predict(const mat &input) {

	uword maxIndex;
	this->feed_forward(input).max(maxIndex);
	return maxIndex;
}

/****************************************************************************************/
/**
 * Saves the trained neural network to a file in ascii format
 * @param name of the file to store the net.
 */
void NeuralNetwork::save(const string &file_name) {

	ofstream writer;
	writer.open(file_name.c_str());

	writer<<num_layers<<"=";
	for(int i = 0; i < num_layers; i++) {

		writer<<layer_sizes[i]<<";";
	}
	writer<<endl;

	for (int i = 0; i < (num_layers - 1); i++) {

		writer<<i<<endl;

		writer<<weights[i].n_rows<<";"<<weights[i].n_cols<<endl;
		for(size_t j = 0; j < weights[i].n_rows; j++) {
			for(size_t k = 0; k < weights[i].n_cols; k++) {

				writer<<weights[i](j, k)<<";";
			}
		}
		writer<<endl;

		writer<<biases[i].n_rows<<";"<<biases[i].n_cols<<endl;
		for (size_t j = 0; j < biases[i].n_rows; j++) {
			for (size_t k = 0; k < biases[i].n_cols; k++) {

				writer << biases[i](j, k) << ";";
			}
		}
		writer<<endl;
	}

	writer.close();
}

/****************************************************************************************************/
/**
 * Loads a trained neural net from a file
 * @param name of the file
 */
void NeuralNetwork::load(const string &file_name) {

	string line;
	ifstream reader(file_name.c_str());
	if (reader.is_open()) {

		//load layer sizes
		load_sizes(reader);
		//load weights and biases for each layer
		for(int i = 0; i < (num_layers - 1); i++) {

			getline(reader, line);
			int layer = atoi(line.c_str());
			if(layer != i)
				break;

			load_parameter_matrix(reader, weights);
			load_parameter_matrix(reader, biases);
		}
	}

	reader.close();
}

/*********************************************************************************************************/
/**
 * private method the load the sizes of the net when is loaded from a file
 * @param stl ifstream reader pointing to the net file.
 */
void NeuralNetwork::load_sizes(ifstream &reader) {

	string line;
	getline(reader, line);
	stringstream ss_layers(line);
	string s_num_layers;
	getline(ss_layers, s_num_layers, '=');
	num_layers = atoi(s_num_layers.c_str());
	layer_sizes = new int[num_layers];
	string s_layer_sizes;
	getline(ss_layers, s_layer_sizes, '=');

	stringstream ss_layer_sizes(s_layer_sizes);
	string token;
	for (int i = 0; i < num_layers; i++) {
		getline(ss_layer_sizes, token, ';');
		layer_sizes[i] = atof(token.c_str());
	}
}

/*****************************************************************************************************************/
/**
 * private method to load a parameter matrix (ex. weights or biases of a given layer) from a file
 * @param stl ifstream reader pointing to the net file.
 * @param stl vector of armadillo matrix where to store the parameters
 */
void NeuralNetwork::load_parameter_matrix(ifstream &reader, vector<mat> &destiny) {

	string line;
	getline(reader, line);
	stringstream ss_dimensions(line);
	string s_rows, s_cols;
	getline(ss_dimensions, s_rows, ';');
	int rows = atoi(s_rows.c_str());
	getline(ss_dimensions, s_cols, ';');
	int cols = atoi(s_cols.c_str());

	mat parameters(rows, cols);

	getline(reader, line);
	stringstream ss_params(line);
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {

			string param;
			getline(ss_params, param, ';');
			parameters(i, j) = atof(param.c_str());
		}
	}

	destiny.push_back(parameters);
}

/*********************************************************************************************/
/**
 * Debug method to print the net to console.
 * It will print sizes and weights/biases matrix for each layer.
 */
void NeuralNetwork::print() {

	cout<<"Net: "<<endl;

	cout<<"num layers: "<<num_layers<<endl<<"sizes: ";
	for (int i = 0; i < num_layers; i++) {

		cout << layer_sizes[i] << ", ";
	}
	cout<<endl;

	for (int i = 0; i < (num_layers - 1); i++) {

		cout << "weights[" << i << "]= " << endl << weights[i] << endl;
		cout << "biases[" << i << "]= " << endl << biases[i] << endl;
	}
}

/**************************************************************************************************/
/**
 * Utility method to visualize what are learning the hidden units in given layer.
 * The method uses OpenCV to show each hidden unit in a window.
 * What is shown is the input for wich the hidden unit is maximally activated.
 * @param number of the layer to visualize.
 * @param size of the image (width and height). The image is assumed to be square.
 */
void NeuralNetwork::visualize_hidden_units(int layer_number, int image_size) {

	for (int i = 0; i < layer_sizes[layer_number]; i++) {

		cv::Mat image(image_size, image_size, 0);

		mat factor = weights[layer_number - 1].row(i) * weights[layer_number - 1].row(i).t();

		int k = 0;
		for (int r = 0; r < image_size; r++) {
			for (int c = 0; c < image_size; c++) {

				image.at<uchar> (r, c) = ((weights[layer_number - 1](i, k++)) / sqrt(factor(0,0))) * 2550000;
			}
		}

		cout<<image<<endl<<endl;
		stringstream ss;
		ss << "[" << i + 1 << "]";
		namedWindow(ss.str());
		imshow(ss.str(), image);
	}

	waitKey(0);
}
/*************************************************************************************************/
/**
 * Saves the monitored data during training to several files in the given directory.
 * The data is in the format of two columns separated by space, the first is the number of epoch, the second is the value of cost/accuracy.
 * This is a useful format to plot the data.
 * @param name of the directory where to save the monitored data files
 */
void NeuralNetwork::save_monitored_data(const string &dir) {

	save_monitored_data_file(dir, "training_costs.dat", training_costs);
	save_monitored_data_file(dir, "training_accuracies.dat", training_accuracies);
	save_monitored_data_file(dir, "validation_costs.dat", validation_costs);
	save_monitored_data_file(dir, "validation_accuracies.dat", validation_accuracies);
}

/*************************************************************************************************/
/**
 * Auxiliar method to save a file with monitored data
 * The data is in the format of two columns separated by space, the first is the number of epoch, the second is the value of cost/accuracy.
 * This is a useful format to plot the data.
 * @param name of the directory where to save the monitored data file
 * @para name of the file
 */
void NeuralNetwork::save_monitored_data_file(const string &dir, const string &file_name, const vector<double> &data) {

	stringstream file;
	file << dir << "/" << file_name;
	ofstream writter(file.str().c_str());

	if(writter.is_open()) {

		for(size_t i = 0; i < data.size(); i++) {

			writter << i+1 << " " << data[i] << endl;
		}
	}

	writter.close();
}

/**************************************************************************************************/
NeuralNetwork::~NeuralNetwork() {
	// TODO Auto-generated destructor stub
}
