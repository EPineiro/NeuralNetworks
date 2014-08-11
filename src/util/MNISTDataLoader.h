/*
 * DataLoader.h
 *
 *  Created on: Aug 8, 2014
 *      Author: epineiro
 */

#ifndef DATALOADER_H_
#define DATALOADER_H_

#include <iostream>
#include <vector>

#include <armadillo>

class MNISTDataLoader {
private:
	static const int cant_classes = 10;
public:
	MNISTDataLoader();
	virtual ~MNISTDataLoader();

	static void load_mnist_data(
			std::vector<std::pair<arma::mat, arma::mat> > &training_mat,
			std::vector<std::pair<arma::mat, arma::mat> > &validation_mat,
			std::vector<std::pair<arma::mat, arma::mat> > &test_mat,
			const std::string &data_dir, int cant_validation_elems);

	static void load_mnist_data(
			std::vector<std::pair<arma::mat, arma::mat> > &data,
			std::string &data_file, std::string &labels_file);

	static void load_reduce_mnist_data(
			std::vector<std::pair<arma::mat, arma::mat> > &training_mat,
			std::vector<std::pair<arma::mat, arma::mat> > &test_mat,
			const std::string &data_dir, int cant_test_elems);

	static int reverse_int(int i);
	static arma::mat vectorize_label(int label);
};

#endif /* DATALOADER_H_ */
