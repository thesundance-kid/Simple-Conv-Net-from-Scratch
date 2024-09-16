import numpy as np

def TwoDim_Conv_layer(inputs, kernels, bias, stride = 1, flattened = True):
    '''
    Take a 2d array (a black and white image), and applys a convultion using kernels (d filters of size (k,k)) using stride specified.
    To take advantage of vectorization, the input and kernel are flattened, and dot products are taken. Output is returned as a flattened layer that 
    can easily be put used to convert into a fully connected dense layer.

    INPUTS:
    input(numpy array): shape must be num of images, height, width -> assuming only black and white images for now, height and width should be equal so 
            shape equal to (m, h, w)
    kernels(numpy array): parameters for kernel used in the conv layer, shape (d, k, k), d is equal to number of kernels/filters
    bias(numpy array): shape (d, 1) one bias value for each filter
    stride(int)

    OUTPUTS:
    flat_output_array(np array): shape (d *o^2, m), 
            each row represents the output layer stacked vertically for all depths, each column is for each input example

    
    '''
    m, h, w = inputs.shape
    k, d, s = kernels.shape[1], kernels.shape[0], stride

    assert (w - k) % s == 0 and (h - k) % s == 0, "Stride or kernel size is invalid"

    # Calculate output dimensions
    output_h = (h - k) // s + 1
    output_w = (w - k) // s + 1

    # Initialize output
    flat_output_array = np.zeros((d * output_h * output_w, m))

    kernels = kernels.reshape((d, k**2))

    out_idx = 0
    for i in range(0, h - k + 1, s):
        for j in range(0, w - k + 1, s):
            patch = inputs[:, i:i+k, j:j+k].reshape(m, -1)
            conv_result = np.dot(kernels, patch.T) + bias
            flat_output_array[out_idx * d:(out_idx + 1) * d, :] = conv_result
            out_idx += 1
    
    if flattened == True:
        return flat_output_array, output_h, output_w
    
    elif flattened == False:
        output_maps = flat_output_array.T.reshape(m, d, output_h, output_w)
        return output_maps
    
def Relu(array):
    return np.maximum(0, array)

def make_params(features, classes):
    """
    Input: 
    Features(int) = number of features in each example thats being passed into fully connected layer
    Classes (int) = how many classes the problem solution has. Will determine how many output neurons from 
    the fully connected layer
    
    Returns: 
    W: (classes, features) radnomly set parameters for W
    b: (classes, 1)
    """
    n_in, n_out = features, classes
    # Compute the standard deviation for the normal distribution
    stddev = np.sqrt(2 / (n_in + n_out))
    # Initialize weights from a normal distribution with mean 0 and stddev
    W = np.random.normal(0, stddev, size=(n_out, n_in))
    b2 = np.zeros((n_out, 1))

    return W, b2

def forward_prop(input, W, b2):
    output = np.dot(W, input) + b2
    return output

def softmax(x):
    """
    Compute the softmax for each column (example) in the input array.
    
    Parameters:
    x (numpy array): The input array with shape (number of classes, number of examples).
    
    Returns:
    numpy array: Softmax applied along the classes axis (axis=0).
    """
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # Subtract max for numerical stability
    return exp_x / exp_x.sum(axis=0, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    """
    Compute the cross-entropy loss.

    Parameters:
    y_true (numpy array): One-hot encoded true labels with shape (number of classes, number of examples).
    y_pred (numpy array): Predicted probabilities from softmax with shape (number of classes, number of examples).

    Returns:
    float: The cross-entropy loss.
    """
    # Clip y_pred to avoid log(0) which results in NaN
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    # Compute the cross-entropy loss
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]
    
    return loss

def clip_gradients(gradients, max_norm):
    if isinstance(gradients, dict):
        # Handle dictionary of gradients
        total_norm = np.sqrt(sum(np.sum(np.square(grad)) for grad in gradients.values()))
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for grad in gradients.values():
                grad *= clip_coef
    elif isinstance(gradients, np.ndarray):
        # Handle single numpy array (e.g., dL_dkernels)
        total_norm = np.sqrt(np.sum(np.square(gradients)))
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            gradients *= clip_coef
    else:
        raise TypeError("gradients must be either a dictionary or a numpy array")
    return gradients

def backprop(learning_rate, y_true, y_pred, A, W, b2, kernels, x, output_h, output_w, stride = 1, clip_grads = True):
    """
    Inputs:
    A: shape (d * o^2, m), flat_output_array after having Relu applied 
    W: shape (num prediction classes, d * o^2), parameters used during forward prop in fully connected layer
    kernels: shape (d, k, k), where d is the number of kernels and k is the kernel size
    x: shape (m, h, w), input images
    """
    m = y_pred.shape[1]
    d, k = kernels.shape[0], kernels.shape[1]

    # Backprop over fully connected layer
    dL_dZf = y_pred - y_true  # shape (num_classes, m)
    dL_dW = np.dot(dL_dZf, A.T) / m  # shape (num_classes, d * output_h * output_w)
    dL_db2 = np.mean(dL_dZf, axis=1, keepdims=True)  # shape (num_classes, 1)
    dL_dA = np.dot(W.T, dL_dZf)  # shape (d * output_h * output_w, m)
    dL_dout = np.where(A > 0, dL_dA, 0)  # shape (d * output_h * output_w, m)

    # Reshape dL_dout for convolution backprop
    dL_dout = dL_dout.T.reshape(m, d, output_h, output_w)  # shape (m, d, output_h, output_w)

    # Backprop over Conv layer
    dL_dkernels = np.zeros_like(kernels)  # shape (d, k, k)
    for i in range(output_h):
        for j in range(output_w):
            input_slice = x[:, i*stride:i*stride+k, j*stride:j*stride+k]  # shape (m, k, k)
            dL_dout_slice = dL_dout[:, :, i, j]  # shape (m, d)
            for f in range(d):
                dL_dkernels[f] += np.sum(input_slice * dL_dout_slice[:, f, np.newaxis, np.newaxis], axis=0)

    #Clip gradients
    if clip_grads == True:
        dL_dW = clip_gradients(dL_dW, max_norm= 1.0)
        dL_db2 = clip_gradients(dL_db2, max_norm= 1.0)
        dL_dkernels = clip_gradients(dL_dkernels, max_norm = 1.0)

    #Update parameters
    W -= learning_rate * dL_dW
    b2 -= learning_rate * dL_db2
    kernels -= learning_rate * dL_dkernels

    return W, b2, kernels





