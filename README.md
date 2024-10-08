# Simple-Conv-Net-from-Scratch
**Simple Convolutional Neural Network (CNN) for Classification**

This repository contains a simple implementation of a Convolutional Neural Network (CNN) in Python (using only numpy), tested to classify handwritten digits from the MNIST dataset. The CNN model consists of one convolutional layer and a softmax output layer, making it ideal for  simple projects.

**Project Structure**

ConvNet_model.py: This file contains the code for building the CNN model. It defines the model structure, including the convolutional layer, ReLU activation, and the softmax layer.
ConvNet_functions.py: Contains the helper functions used in the model, such as:

TwoDim_Conv_layer: Applies a 2D convolution to the input image using the provided filters (kernels).
Relu: Implements the ReLU activation function.
make_params: Initializes the parameters for the model.
forward_prop: Conducts forward propagation through the network.
softmax: Implements the softmax function for the output layer.
cross_entropy_loss: Calculates the cross-entropy loss.
backprop: Handles backpropagation for all layers, and update the weights based on the loss.

test.ipynb: This Jupyter Notebook demonstrates how to load the MNIST dataset, preprocess the images and labels, and train the CNN model using the functions provided in the other files. The notebook also includes an accuracy assessment on the test data.

**Model Description**

The CNN implemented in this project is straightforward and consists of:

Convolutional Layer: A 2D convolution is applied to the input images using a small kernel. This layer extracts features from the images such as edges and textures.
ReLU Activation: The output of the convolutional layer is passed through a ReLU activation function to introduce non-linearity.
Softmax Output Layer: The final layer is a fully connected dense layer followed by a softmax layer that classifies the input images into 10 classes (digits 0-9).

**Training Process**
The network is trained on the MNIST dataset using a forward propagation approach followed by backpropagation.
The dataset is preprocessed by normalizing the pixel values and one-hot encoding the labels.
Cross-entropy loss is used to measure the error between the predicted output and the actual labels, and backpropagation updates the parameters to minimize this loss.
Time to train for 100 epochs on the full 6000 image training set was around 5 hours on my personal macbook. 
Time to train for 100 epochs on just 2000 images, was only about 20 minutes. The accuracy however did not surpass around 85%.

**Results**
The model was trained on the full MNIST dataset (6000 images). Here are some performance metrics:

Test Accuracy: ~93.5%
The notebook provides a more detailed view of the training process and evaluation of the model's performance.


MNIST Dataset
The MNIST dataset must be downloaded separately if you haven't already. You can find the dataset here: https://www.kaggle.com/datasets/hojjatk/mnist-dataset
Ensure that the dataset is placed in the correct path referenced in the notebook (./archive/train-images.idx3-ubyte and ./archive/train-labels.idx1-ubyte).

