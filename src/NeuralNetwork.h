/*
 * NeuralNetwork.h
 *
 *  Created on: Jul 25, 2014
 *      Author: epineiro
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iterator>

#include <armadillo>
#include <opencv2/opencv.hpp>

typedef struct {

	int cant_epochs;
	int mini_batch_size;
	double eta;
	double lambda;
	double mu;
	double beta;
	double sparsity_param;

} TrainingParams;

class NeuralNetwork {

private:
	int num_layers;
	int *layer_sizes;
	std::vector<arma::mat> biases;
	std::vector<arma::mat> weights;

	std::vector<arma::mat> momentum_biases;
	std::vector<arma::mat> momentum_weights;

	bool monitor_training_cost;
	bool monitor_training_accuracy;
	bool monitor_validation_cost;
	bool monitor_validation_accuracy;

	std::vector<double> training_costs;
	std::vector<double> training_accuracies;
	std::vector<double> validation_costs;
	std::vector<double> validation_accuracies;

	inline arma::mat sigmoid(const arma::mat &z) {
		return  1.0 / (1.0 + exp(-z));
	}

	inline arma::mat sigmoid_prime(const arma::mat &z) {
		return sigmoid(z) % (1.0 - sigmoid(z));
	}

	void update_mini_batch(
			std::vector<std::pair<arma::mat, arma::mat> >::const_iterator begin,
			std::vector<std::pair<arma::mat, arma::mat> >::const_iterator end,
			TrainingParams params, int training_size);

	void back_propagation(const arma::mat &x, const arma::mat &y,
			std::vector<arma::mat> &delta_nabla_w,
			std::vector<arma::mat> &delta_nabla_b);


	void load_sizes(std::ifstream &reade);
	void load_parameter_matrix(std::ifstream &reader, std::vector<arma::mat> &destiny);
	void save_monitored_data_file(const std::string &dir, const std::string &file_name, const std::vector<double> &data);

public:
	NeuralNetwork(){}
	NeuralNetwork(int sizes[], int num_layers);
	virtual ~NeuralNetwork();

	arma::mat feed_forward(const arma::mat &input);

	void stochastic_gradient_descent_training(
			std::vector<std::pair<arma::mat, arma::mat> > &data, TrainingParams params,
			std::vector<std::pair<arma::mat, arma::mat> > test_data = std::vector<std::pair<arma::mat, arma::mat> >());

	int evaluate(const std::vector<std::pair<arma::mat, arma::mat> > &test_data);
	int predict(const arma::mat &input);

	double calcule_total_cost(std::vector<std::pair<arma::mat, arma::mat> > &data, double lambda);

	void save(const std::string &file_name);
	void load(const std::string &file_name);
	void print();
	void visualize_hidden_units(int layer_number, int image_size);

	void save_monitored_data(const std::string &dir);

	/**
	 * Methods used to indicate if the network should calcule costs and accuracies for the data sets during training.
	 * This data is stored later in files that can be analized
	 */
    inline void set_monitor_training_accuracy(bool monitor_training_accuracy) {
		this->monitor_training_accuracy = monitor_training_accuracy;
	}

	inline void set_monitor_training_cost(bool monitor_training_cost) {
		this->monitor_training_cost = monitor_training_cost;
	}

	inline void set_monitor_validation_accuracy(bool monitor_validation_accuracy) {
		this->monitor_validation_accuracy = monitor_validation_accuracy;
	}

	inline void set_monitor_validation_cost(bool monitor_validation_cost) {
		this->monitor_validation_cost = monitor_validation_cost;
	}

};

#endif /* NEURALNETWORK_H_ */
