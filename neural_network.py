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

        self.network_weights_delta = []
        self.network_weights_delta_transposed = []
        self.network_weights_bias_delta = []

        self.network_layers = []
        self.network_layers_activation = []
        self.network_layers_error = []

        self.network_input = None

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

        for i in range(self.network_layers_count - 1):
            self.network_weights_bias[i] = numpy.matrix(numpy.random.normal(size = self.network_weights_bias[i].shape[0]))


    # calculate output values
    def propagate_forward(self, input):

        self.network_input = input
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

    def calculate_delta(self):

        self.network_weights_delta[0] = numpy.outer(self.network_input, self.network_layers_error[0])
        self.network_weights_delta_transposed[0] = numpy.transpose(self.network_weights_delta[0])
        self.network_weights_bias_delta[0] = self.network_layers_error[0]

        for i in range(1, self.network_layers_count - 1):
            self.network_weights_delta[i] = numpy.outer(self.network_layers[i], self.network_layers_error[i])
            self.network_weights_delta_transposed[i] = numpy.transpose(self.network_weights_delta[i])
            self.network_weights_bias_delta[i] = self.network_layers_error[i]

    def update_weights(self):

        for i in range(0, self.network_layers_count - 1):
            self.network_weights[i] -= self.network_weights_delta[i]
            self.network_weights_transposed[i] -= self.network_weights_delta[i]
            self.network_weights_bias[i] -= self.network_weights_bias_delta[i]



    class neural_network_test:

        def run_basic_test(self):

            input_size = 4
            nn = neural_network([input_size, 2])
            input = numpy.matrix(numpy.random.normal(size=input_size))
            output = nn.propagate_forward(input)

            #TODO: check formulae manually


        def run_gradient_test(self):

            print "Starting gradient test..."

            input_height = 28
            input_width = 28
            input_size = input_width * input_height
            output_classes = 10
            network_layers = [input_size, 64, 32, output_classes]
            epsilon = 0.0005

            print "Neural network layers: {}".format(network_layers)
            print "Gradient epsilon: {}".format(epsilon)

            nn = neural_network(network_layers)

            input = numpy.matrix(numpy.random.normal(size=input_size))
            print "Input: {}".format(input)

            output = nn.propagate_forward(input)
            print "Output: {}".format(output)

            error = numpy.ones((1, output_classes))
            nn.propagate_backward(error)

            total = 0
            passed = 0

            # compute gradient with respect to i-th input
            input_gradient = numpy.matrix(numpy.zeros(input.shape))
            for i in range(input_gradient.shape[1]):

                # compute gradient df = (f(x+h) - f(x-h)) / h
                h = 0.0005
                x = input[(0, i)]

                input[0, i] = x - h
                output_a = nn.propagate_forward(input)

                input[0, i] = x + h
                output_b = nn.propagate_forward(input)

                gradient = (output_b - output_a) / (2 * h)

                input[0, i] = x

                for j in range(nn.network_layers[0].shape[1]):
                    input_gradient[0, i] += nn.network_weights[0][i, j] * nn.network_layers_error[0][0, j]

                delta = abs(input_gradient[0, j] - numpy.sum(gradient))

                if delta < epsilon:
                    print "Test case for {}-th input failed: Calculated = {} Estimated = {} Delta = {}".format(i, input_gradient[0, j], numpy.sum(gradient), delta)
                else:
                    passed += 1
                    print "Test case for {}-th input passed".format(i)

            print "Passed {} tests from {}".format(passed, input_gradient.shape[1])


test = neural_network.neural_network_test()
test.run_gradient_test()