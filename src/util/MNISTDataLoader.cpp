/*
 * DataLoader.cpp
 *
 *  Created on: Aug 8, 2014
 *      Author: epineiro
 */

#include "MNISTDataLoader.h"

using namespace std;
using namespace arma;

MNISTDataLoader::MNISTDataLoader() {
	// TODO Auto-generated constructor stub

}

MNISTDataLoader::~MNISTDataLoader() {
	// TODO Auto-generated destructor stub
}

void MNISTDataLoader::load_mnist_data(vector<pair<mat, mat> > &training_mat,
		vector<pair<mat, mat> > &validation_mat,
		vector<pair<mat, mat> > &test_mat, const string &data_dir,
		int cant_validation_elems) {

	//
	cout << "loading training data" << endl;
	string train_data_file(data_dir), train_labels_file(data_dir);
	train_data_file.append("/train-images.idx3-ubyte");
	train_labels_file.append("/train-labels.idx1-ubyte");
	load_mnist_data(training_mat, train_data_file, train_labels_file);
	//we will only use the first 50000 examples to train and the last 10000 to validate hyperparameters
	validation_mat.insert(validation_mat.begin(), training_mat.end() - cant_validation_elems, training_mat.end());
	training_mat.erase(training_mat.end() - cant_validation_elems, training_mat.end());
	cout << "training data loaded" << endl;
	//*/

	cout << "loading test data" << endl;
	string test_data_file(data_dir), test_labels_file(data_dir);
	test_data_file.append("/t10k-images.idx3-ubyte");
	test_labels_file.append("/t10k-labels.idx1-ubyte");
	load_mnist_data(test_mat, test_data_file, test_labels_file);
	//just for test, we will only use the first 5000 examples to test
	//test_mat.erase(test_mat.end() - CANT_TEST_ELEM, test_mat.end());
	cout << "test data loaded" << endl;
}

void MNISTDataLoader::load_mnist_data(vector<pair<mat, mat> > &data, string &data_file, string &labels_file) {

	ifstream data_reader(data_file.c_str(), ios::binary);
	ifstream labels_reader(labels_file.c_str(), ios::binary);

	if (data_reader.is_open() && labels_reader.is_open()) {

		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		data_reader.read((char*) &magic_number, sizeof(magic_number));
		magic_number = reverse_int(magic_number);
		data_reader.read((char*) &number_of_images, sizeof(number_of_images));
		number_of_images = reverse_int(number_of_images);
		data_reader.read((char*) &n_rows, sizeof(n_rows));
		n_rows = reverse_int(n_rows);
		data_reader.read((char*) &n_cols, sizeof(n_cols));
		n_cols = reverse_int(n_cols);

		//we read again this parameters from the label file,
		//just to advance the pointer in the file.
		//the magic number is not used and the number of image should be the same
		labels_reader.read((char*) &magic_number, sizeof(magic_number));
		magic_number = reverse_int(magic_number);
		labels_reader.read((char*) &number_of_images, sizeof(number_of_images));
		number_of_images = reverse_int(number_of_images);

		for (int i = 0; i < number_of_images; i++) {

			unsigned char temp = 0;

			//read image
			mat tp(n_rows * n_cols, 1);
			for (int r = 0; r < n_rows * n_cols; r++) {

				data_reader.read((char*) &temp, sizeof(temp));
				tp(r, 0) = (double) temp;
			}

			//read label
			labels_reader.read((char*) &temp, sizeof(temp));
			mat label = vectorize_label((int)temp);

			data.push_back(make_pair(tp, label));
		}
	}

	data_reader.close();
	labels_reader.close();
}

int MNISTDataLoader::reverse_int(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

mat MNISTDataLoader::vectorize_label(int label) {

	mat result = zeros(cant_classes, 1);
	result(label) = 1;

	return result;
}

void MNISTDataLoader::load_reduce_mnist_data(vector<pair<mat, mat> > &training_mat,
		vector<pair<mat, mat> > &test_mat, const string &data_dir, int cant_test_elems) {

	mat X, Y;

	cout<<"loading training examples."<<endl;

	string x_file(data_dir);
	x_file.append("/X.dat");
	X.load(x_file, raw_ascii);

	cout<<"X = num rows: "<<X.n_rows<<" num cols: "<<X.n_cols<<endl;

	cout<<"loading training labels."<<endl;

	string y_file(data_dir);
	y_file.append("/Y.dat");
	Y.load(y_file, raw_ascii);

	cout<<"Y = num rows: "<<Y.n_rows<<" num cols: "<<Y.n_cols<<endl;

	for(size_t i = 0; i < X.n_rows; i++) {
		//both the example and the label are stored as colum vector
		training_mat.push_back(make_pair(X.row(i).t(), vectorize_label(Y(i))));
	}

	//we divide the data into training and test
	random_shuffle(training_mat.begin(), training_mat.end());
	test_mat.insert(test_mat.begin(), training_mat.end() - cant_test_elems, training_mat.end());
	training_mat.erase(training_mat.end() - cant_test_elems, training_mat.end());

	cout<<"training data loaded"<<endl;
}
