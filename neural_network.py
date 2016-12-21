import numpy

def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))

# gradient of sigmoid
def sigmoid_gradient(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

network_layers_count = 4

# input_depth = 3
input_height = 32
input_width = 32
input_size = input_width * input_height
output_classes = 16

network_layers_size = [input_size, 64, 32, output_classes]

# build network
network_weights = []
network_weights_bias = []
network_layers = []
network_layers_activation = []

# delta for layer
network_layers_delta = []
network_layers_error = []

for i in range(network_layers_count-1):
    network_layers.append(numpy.matrix(numpy.zeros(network_layers_size[i+1])))
    network_layers_activation.append(numpy.matrix(numpy.zeros(network_layers_size[i+1])))
    network_layers_delta.append(numpy.matrix(numpy.zeros(network_layers_size[i+1])))
    network_layers_error.append(numpy.zeros(network_layers_size[i+1]))

for i in range(network_layers_count-1):
    network_weights_shape = (network_layers_size[i], network_layers_size[i+1])
    network_weights.append(numpy.matrix(numpy.zeros(network_weights_shape)))

for i in range(network_layers_count-1):
    network_weights_bias.append(numpy.matrix(numpy.zeros(network_layers_size[i+1])))

for i in range(len(network_layers)):
    print "Network layer {}-th size: {}".format(i, network_layers[i].shape)

for i in range(len(network_weights)):
    print "Network weights {}-th size: {}".format(i, network_weights[i].shape)

for i in range(len(network_weights_bias)):
    print "Network bias {}-th size: {}".format(i, network_weights_bias[i].shape)



# calculate output values
def propagate_forward(x):

    network_layers_activation[0] = x * network_weights[0]
    network_layers_activation[0] += network_weights_bias[0]
    network_layers[0] = sigmoid(network_layers_activation[0])

    for i in range(1, network_layers_count-1):
        network_layers_activation[i] = network_layers[i-1] * network_weights[i]
        network_layers_activation[i] += network_weights_bias[i]
        network_layers[i] = sigmoid(network_layers_activation[i])

    return network_layers[-1] # last element


# initialize randomly
for i in range(len(network_weights)):
    network_weights_shape = (network_weights[i].shape[0], network_weights[i].shape[1])
    network_weights[i] = numpy.matrix(numpy.random.normal(size = network_weights_shape))

#for i in range(len(network_weights_bias)):
#    network_weights_bias[i] = numpy.matrix(numpy.random.normal(size = network_weights_bias[i].shape[0]))

input = numpy.matrix(numpy.random.normal(size = input_size))
print "Input\n {}".format(input)

output = propagate_forward(input)
print "Output\n {}".format(output)


#network_weights[0][0, 0] = 0.2
#network_weights[1][0, 0] = 0.3

#input[0, 0] = 1.0

# set last error to delta

errors = numpy.ones(network_layers_error[-1].shape)
network_layers_error[-1] = errors * numpy.asarray(sigmoid_gradient(network_layers_activation[-1]))

# compute gradient
for i in range(len(network_layers)-2, -1, -1):
    network_layers_error[i] = numpy.zeros(network_layers_error[i].shape)

    for j in range(network_layers[i].shape[1]):
        for k in range(network_layers[i+1].shape[1]):
            network_layers_error[i][j] += network_weights[i+1][j, k] * network_layers_error[i+1][k]

    network_layers_error[i] = network_layers_error[i] * numpy.asarray(sigmoid_gradient(network_layers_activation[i]))

# compute gradient with respect to input
input_error = numpy.matrix(numpy.zeros(input.shape))
for j in range(input_error.shape[1]):

    # compute gradient df = (f(x+h) - f(x-h)) / h

    h = 0.0005
    x = input[(0, j)]

    input[0, j] = x - h
    output_a = propagate_forward(input)

    input[0, j] = x + h
    output_b = propagate_forward(input)

    delta = (output_b - output_a)
    gradient = delta / (2 * h)

    input[0, j] = x
    output = propagate_forward(input)

    for k in range(network_layers[0].shape[1]):
        input_error[0, j] += network_weights[0][j, k] * network_layers_error[0][0, k] * sigmoid_gradient(network_layers_activation[0][0, k])

    print "Gradient for {}-th input: \n {}".format(j, input_error[0, j])
    print "Gradient estimate for {}-th input: \n {}".format(j, numpy.sum(gradient))

    print "\n"