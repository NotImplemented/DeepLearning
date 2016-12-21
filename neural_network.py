import numpy

class neural_network:

    # activation function
    def sigmoid(self, x):
        return 1.0 / (1.0 + numpy.exp(-x))

    # gradient of sigmoid
    def sigmoid_gradient(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))


    # builds network using sizes of layers
    def __init__(self, layers):

        self.network_layers_size = layers
        self.network_layers_count = len(self.network_layers_size)

        # create network
        self.network_weights = []
        self.network_weights_transposed = []
        self.network_weights_bias = []

        self.network_layers = []
        self.network_layers_activation = []
        self.network_layers_error = []

        for i in range(self.network_layers_count - 1):
            self.network_layers.append(None)
            self.network_layers_activation.append(None)
            self.network_layers_error.append(None)

        for i in range(self.network_layers_count - 1):
            network_weights_shape = (self.network_layers_size[i], self.network_layers_size[i + 1])
            self.network_weights.append(numpy.matrix(numpy.random.normal(size=network_weights_shape)))
            self.network_weights_transposed.append(self.network_weights[i].transpose())

        for i in range(self.network_layers_count - 1):
            self.network_weights_bias.append(numpy.matrix(numpy.zeros(self.network_layers_size[i + 1])))

        # for i in range(len(network_weights_bias)):
        #    network_weights_bias[i] = numpy.matrix(numpy.random.normal(size = network_weights_bias[i].shape[0]))


    # calculate output values
    def propagate_forward(self, input):

        self.network_layers_activation[0] = input * self.network_weights[0]
        self.network_layers_activation[0] += self.network_weights_bias[0]
        self.network_layers[0] = self.sigmoid(self.network_layers_activation[0])

        for i in range(1, self.network_layers_count - 1):
            self.network_layers_activation[i] = self.network_layers[i - 1] * self.network_weights[i]
            self.network_layers_activation[i] += self.network_weights_bias[i]
            self.network_layers[i] = self.sigmoid(self.network_layers_activation[i])

        return self.network_layers[-1]  # last element

    # calculate output values
    def propagate_backward(self, error):

        # compute gradient explicitly
        # for i in range(len(network_layers)-2, -1, -1):
        #    network_layers_error[i] = numpy.zeros(network_layers[i].shape)

        #    for j in range(network_layers[i].shape[1]):
        #        for k in range(network_layers[i+1].shape[1]):
        #            network_layers_error[i][(0, j)] += network_weights[i+1][j, k] * network_layers_error[i+1][(0, k)]

        #    network_layers_error[i] *= sigmoid_gradient(numpy.array(network_layers_activation[i]))

        # compute gradient using matrix multiplication
        self.network_layers_error[-1] = error * self.sigmoid_gradient(numpy.array(self.network_layers_activation[-1]))

        for i in range(self.network_layers_count - 3, -1, -1):
            self.network_layers_error[i] = self.network_layers_error[i + 1] * self.network_weights_transposed[i + 1]
            self.network_layers_error[i] = numpy.asarray(self.network_layers_error[i]) * self.sigmoid_gradient(numpy.array(self.network_layers_activation[i]))






# main starts here

input_height = 32
input_width = 32
input_size = input_width * input_height
output_classes = 10

nn = neural_network([input_size, 64, 32, output_classes])


input = numpy.matrix(numpy.random.normal(size = input_size))
print "Input\n {}".format(input)

output = nn.propagate_forward(input)
print "Output\n {}".format(output)

error = numpy.ones((1, output_classes))
nn.propagate_backward(error)


# compute gradient with respect to input
input_gradient = numpy.matrix(numpy.zeros(input.shape))
for j in range(input_gradient.shape[1]):

    # compute gradient df = (f(x+h) - f(x-h)) / h
    h = 0.0005
    x = input[(0, j)]

    input[0, j] = x - h
    output_a = nn.propagate_forward(input)

    input[0, j] = x + h
    output_b = nn.propagate_forward(input)

    delta = (output_b - output_a)
    gradient = delta / (2 * h)

    input[0, j] = x
    output = nn.propagate_forward(input)

    for k in range(nn.network_layers[0].shape[1]):
        input_gradient[0, j] += nn.network_weights[0][j, k] * nn.network_layers_error[0][0, k]

    print "Calculated gradient for {}-th input: \n {}".format(j, input_gradient[0, j])
    print "Gradient estimate for {}-th input: \n {}".format(j, numpy.sum(gradient))

    print "\n"

