import numpy as np

"""
The following is a function definition of the sigmoid function, which is the type of non-linearity chosen for this neural net.
In practice, large-scale deep learning systems use piecewise-linear functions because they are much less expensive to evaluate.

The implementation of this function does double duty. If the deriv=True flag is passed in, the function instead calculates the 
derivative of the function, which is used in the error backpropogation step.
"""

def activFunction(x, deriv=False):
    #Sigmoid
    if (deriv == True):
        return (x * (1 - x))

    return 1 / (1 + np.exp(-x))


def buildANN():
    """
    The following code creates the input matrix. The first and second collumn repreesent the two neurons.
    The third column is for accommodating the bias term and is not part of the input.
    Let's do all the four possible training examples
    It needs to be a List of Lists
    """
    X = np.array([  [0,0,1],
                    [0,1,1],
                    [1,0,1],
                    [1,1,1] ])

    """
    The output of the exclusive OR function follows.
    It needs to be a List of Lists
    """
    Y = np.array([[0],
                 [1],
                 [1],
                 [0]])

    """
    The seed for the random generator is set so that it will return the same random numbers each time, which is sometimes 
    useful for DEBUGGING.  Since we'll be generating random numbers in a second, let's seed them to make them deterministic.
    This just means give random numbers that are generated the same starting point or seed so that we'll get the same sequence 
    of generated numbers every time we run our program.
    """
    np.random.seed(1) #???


    """
    Next, we'll create our synapse matrices. Synapses are the connections between each neuron in one layer to every neuron in the next layer.
    Since we will have three layers in our network, we need two synapses MATRICES. Each synapse has a random weight assigned to it.
    
    So we intialize the weights to random values. syn0 are the weights between the input layer and the hidden layer. 
    It is a 3x4 matrix because there are two input weights plus a bias term (=3) and four nodes in the hidden layer (=4). 
    syn1 are the weights between the hidden layer and the output layer. It is a 4x1 matrix because there are 4 nodes in 
    the hidden layer and one output. Note that there is no bias term feeding the output layer in this example. The weights 
    are initially generated randomly because optimization tends not to work well when all the weights start at the same value. 
    Note that neither of the neural networks shown in the video describe the example.
    """
    syn0 = 2*np.random.random((3,4)) - 1  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
    syn1 = 2*np.random.random((4,1)) - 1  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.

    return (X, Y, syn0, syn1)


"""
This is the main training loop.The output shows the evolution of the error between the model and desired.The error steadily decreases.
"""
def training(X, Y, syn0, syn1):
    # Python2 Note: In the follow command, you may improve
    #   performance by replacing 'range' with 'xrange'.
    for j in range(60000):

        #first layer
        l0 = X

        #second layer
        l1 = activFunction(np.dot(l0, syn0))    # Now comes the prediction step
                                                # matrix multiplication between each layer and its synapse, then we'll run
                                                # our sigmoid function on all the values in the matrix to create

        #third layer
        l2 = activFunction(np.dot(l1, syn1))    # layer that contains a prediction of the output data, which is a more refined prediction.

        # Back propagation of errors using the chain rule.
        # let's compare the output with the expected output data using subtraction to get the error rate.
        l2_error = Y - l2
        if (j % 10000) == 0:  # Only print the average error rate every 10000 steps, to save time and limit the amount of output.
            print("Error: " + str(np.mean(np.abs(l2_error))))

        #BACK-PROPAGATION
        l2_delta = l2_error * activFunction(l2, deriv=True) # Next, we'll multiply our error rate by the result of our sigmoid function.
                                                            # The function is used to get the derivative of our output prediction from layer two.
                                                            # This will give us a delta which we'll use to reduce the error rate of our predictions
                                                            # when we update our synapses every iteration.

        l1_error = l2_delta.dot(syn1.T)                     # Then we'll want to see how much layer one contributed to the error in layer two (BACK-PROPAGATION)
                                                            # We'll get this error by multiplying layer two's delta by synapse one's transpose.

        l1_delta = l1_error * activFunction(l1, deriv=True) # Then we'll get layer one's delta by multiplying its error by the result of our sigmoid function.
                                                            # The function is used to get the derivative of layer one.

        # update weights (no learning rate term) using GRADIENT DESCENT
        # Now that we have deltas for each of our layers, we can use them to update our synapse rates to reduce the error rate more and more every iteration.
        # To do this, we'll just multiply each layer by a delta.
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)

    return l2


def main():
    ANN = buildANN()
    output = training(ANN[0], ANN[1], ANN[2], ANN[3])
    print(output)
    """
    See how the final output closely approximates the true output [0, 1, 1, 0]. If you increase the number of 
    interations in the training loop (currently 60000), the final output will be even closer.
    """


if __name__ == "__main__":
    main()

