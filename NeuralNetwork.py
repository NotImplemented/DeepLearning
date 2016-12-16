import math
import numpy

#activation function
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

#gradient of activation
def sigmoid_gradient(x):
    return 1.0 / (1.0 - x)

network_layers_count = 4

input_height = 32
input_width = 32
input_depth = 3
output_classes = 10

network_layers_size = [input_depth * input_width, 64, 32, output_classes]

# build network
network_weights = []
network_weights_bias = []
network_layers = []

for i in range(network_layers_count):
    network_layers.append(numpy.zeros(network_layers_size[i]))

for i in range(network_layers_count-1):
    network_weights.append(numpy.zeros((network_layers_size[i], network_layers_size[i+1])))

for i in range(network_layers_count-1):
    network_weights_bias.append(numpy.zeros(network_layers_size[i+1]))

# calculate output values
def propagate_forward(x):

    network_layers[0] =  numpy.multiply(x, network_weights[0])
    numpy.add(network_layers[0], network_weights_bias[0], network_layers[0])

    for i in range(1, network_layers_count):
        network_layers[i] = numpy.multiply(network_layers[i-1], network_weights[i])
        numpy.add(network_layers[i], network_weights_bias[i], network_layers[i])

    return network_layers[-1] # last element


# initialize randomly
for i in range(network_layers_count):
    network_layers[i] = numpy.random.uniform(-1, 1, network_layers[i].shape)

for i in range(network_layers_count-1):
    network_weights[i] = numpy.random.uniform(-1, 1, network_weights[i].shape)

for i in range(0,network_layers_count-1):
    network_weights_bias[i] = numpy.random.uniform(-1, 1, network_weights_bias[i].shape)




print network_layers
