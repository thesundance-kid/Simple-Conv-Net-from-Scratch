import numpy as np 
from ConvNet_functions import TwoDim_Conv_layer, Relu, make_params, forward_prop, softmax,  cross_entropy_loss, backprop

def create_cnn_model(input_shape, num_classes, num_filters, kernel_size, stride):
    # Initialize parameters
    kernels = np.random.randn(num_filters, kernel_size, kernel_size) * 0.01
    bias = np.zeros((num_filters, 1))
    
    # Calculate the output shape after convolution
    output_h = (input_shape[1] - kernel_size) // stride + 1
    output_w = (input_shape[2] - kernel_size) // stride + 1
    fc_input_size = num_filters * output_h * output_w
    
    # Initialize fully connected layer parameters
    W, b2 = make_params(fc_input_size, num_classes)
    
    return kernels, bias, W, b2

def forward_pass(x, kernels, bias, W, b2, stride):
    # Convolutional layer
    conv_output, output_h, output_w = TwoDim_Conv_layer(x, kernels, bias, stride, flattened=True)
    
    # ReLU activation
    relu_output = Relu(conv_output)
    
    # Fully connected layer
    fc_output = forward_prop(relu_output, W, b2)
    
    # Softmax
    y_pred = softmax(fc_output)
    
    return y_pred, relu_output, output_h, output_w

def train_cnn(X, y, num_epochs, learning_rate, num_filters, kernel_size, stride):
    '''
    Inputs:
    X: Input images, array with shape (num examples, height, width), for now must be grayscale images with no depth/ a 2d array
    y: True labels, array with shape (num classes, num examples)
    '''
    input_shape = X.shape
    num_classes = y.shape[0]
    
    # Initialize model
    kernels, bias, W, b2 = create_cnn_model(input_shape, num_classes, num_filters, kernel_size, stride)
    
    for epoch in range(num_epochs):
        # Forward pass
        y_pred, relu_output, output_h, output_w = forward_pass(X, kernels, bias, W, b2, stride)
        
        # Compute loss
        loss = cross_entropy_loss(y, y_pred)
        
        # Backpropagation
        W, b2, kernels = backprop(learning_rate, y, y_pred, relu_output, W, b2, kernels, X, output_h, output_w, stride, clip_grads= True)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return kernels, bias, W, b2

def train_cnn_minibatch(X, y, num_epochs, learning_rate, num_filters, kernel_size, stride, batch_size=32):
    '''
    Inputs:
    X: Input images, array with shape (num examples, height, width), for now must be grayscale images with no depth/ a 2d array
    y: True labels, array with shape (num classes, num examples)
    batch_size: Size of mini-batches
    '''
    input_shape = X.shape
    num_classes = y.shape[0]
    num_examples = X.shape[0]
    
    # Initialize model
    kernels, bias, W, b2 = create_cnn_model(input_shape, num_classes, num_filters, kernel_size, stride)
    
    for epoch in range(num_epochs):
        # Shuffle the data
        permutation = np.random.permutation(num_examples)
        X_shuffled = X[permutation]
        y_shuffled = y[:, permutation]
        
        # Mini-batch training
        for i in range(0, num_examples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[:, i:i+batch_size]
            
            # Forward pass
            y_pred, relu_output, output_h, output_w = forward_pass(X_batch, kernels, bias, W, b2, stride)
            
            # Compute loss
            loss = cross_entropy_loss(y_batch, y_pred)
            
            # Backpropagation
            W, b2, kernels = backprop(learning_rate, y_batch, y_pred, relu_output, W, b2, kernels, X_batch, output_h, output_w, stride, clip_grads= True)
        
        if epoch % 10 == 0:
            # Compute loss on full dataset for logging
            y_pred_full, _, _, _ = forward_pass(X, kernels, bias, W, b2, stride)
            loss_full = cross_entropy_loss(y, y_pred_full)
            print(f"Epoch {epoch}, Loss: {loss_full}")
    
    return kernels, bias, W, b2


# Example usage
input_shape = (100, 28, 28)  # 100 images of 28x28 pixels
num_classes = 10
num_filters = 32
kernel_size = 3
stride = 1
num_epochs = 100
learning_rate = 0.01

# Generate dummy data
X = np.random.randn(*input_shape)
y = np.eye(num_classes)[:, np.random.randint(0, num_classes, input_shape[0])]

# Train the model
kernels, bias, W, b2 = train_cnn(X, y, num_epochs, learning_rate, num_filters, kernel_size, stride)

# Make predictions
y_pred, _, _, _ = forward_pass(X, kernels, bias, W, b2, stride)
predicted_classes = np.argmax(y_pred, axis=0)