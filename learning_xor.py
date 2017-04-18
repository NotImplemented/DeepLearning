import numpy
import random
from plot import prediction_error_plot
from neural_network import neural_network
from mnist_database import mnist_database


train_set = [ ((0, 0), 0), ((1, 1), 0), ((0, 1), 1), ((1, 0), 1) ]

def prepare_train_set(train_set):

    train_data_set = []

    for i in range(len(train_set)):
        ((x, y), target) = train_set[i]
        image = numpy.zeros((1,2))

        image[(0, 0)] = x
        image[(0, 1)] = y

        train_data_set.append((image, target))

    return train_data_set

def prediction_error(train_set, nn):

    error = 0;
    for i in range(len(train_set)):
        ((x, y), target) = train_set[i]
        image = numpy.zeros((1,2))

        image[(0, 0)] = x
        image[(0, 1)] = y

        train_data_set.append((image, target))
        output = nn.propagate_forward(image)
        error += (output-target)*(output-target)
        print('Input:', image)
        print('Output:', output)

    print('Error:', error)

def nn_debug(nn):

    print('Weights', nn.network_weights)
    print('Weights bias', nn.network_weights_bias)

train_data_set = prepare_train_set(train_set)
random.shuffle(train_data_set)

# create neural network and set up parameters
learning_rate = 0.35
momentum = 0.04
network_layers = [2, 2, 1]

epochs = 1024 * 256 * 16

print('Creating neural network with layers {}'.format(network_layers))
nn = neural_network(network_layers)


numpy.set_printoptions(precision = 6)

for epoch in range(epochs):

    processed_train_image = 0
    processed_train_image_error = 0
    processed_error = 0.0
    processed_batch = 0

    batch_size = len(train_data_set)
    for i in range(batch_size):

        image = train_data_set[i][0]
        label = train_data_set[i][1]
        output = nn.propagate_forward(image)

        correct = numpy.zeros((1, 1))
        correct[(0,0)] = label

        processed_train_image += 1

        # calculate error function
        error_delta = numpy.asarray(output - correct)
        error = error_delta / batch_size * learning_rate

        nn.propagate_backward(error)
        nn.update_delta()

        processed_error += numpy.sum(numpy.abs(error_delta * error_delta))

    nn.update_weights(learning_rate, momentum)
    nn.reset_weights_delta()
    processed_batch += 1

    if epoch % 1024 == 0:
        print('')
        print('Training epoch #{} was completed.'.format(epoch + 1))
        print('Batch total error: {:.6f}'.format(processed_error / batch_size))
        print('Decaying learning rate: from {:.6f} to {:.6f}'.format(learning_rate, learning_rate * 0.99))
        learning_rate = learning_rate * 0.99
        prediction_error(train_set, nn)
        nn_debug(nn)

    print

print('Training was completed successfully')