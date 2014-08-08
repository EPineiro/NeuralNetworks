NeuralNetworks
==============

- Repository for experimentation with artificial neural networks

- Implementation of Neural Networks to experiment with them.

- In the first version there is just a shallow network trained with stochastic gradient descent with momentum.
- Using a learning rate of 0.001, regularization parameter of 0.1, momentum coefficient of 0.1, 30 hidden neurons and trained for 30 epochs with a mini batch size of 10 I get aprox. 91-92% of accuracy on MNIST dataset

- Most of the code in based on the book "Neural Networks and Deep Learning" by Michael A. Nielsen, but translated to C++ (the original code is in python)

- The external libraries that the project use are:
-   Armadillo: This is a library for linear algebra. Most of the operations are wrapper to LAPACK/BLAS, so you need to have this two libraries in your system too.
-   OpenCV: currently is only used to visualize images (like the training data)
