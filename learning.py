import numpy
from plot import prediction_error_plot
from neural_network import neural_network
from mnist_database import mnist_database
from random import shuffle

database = mnist_database()

# prepare training data set
images = database.get_train_images()
labels = numpy.frombuffer(database.get_train_labels(), dtype = numpy.dtype(numpy.uint8))

image_height = database.image_height
image_width = database.image_width
output_classes = 10

if len(images) != len(labels):
    raise ValueError('Train data set mismatch: {} labels, {} images'.format(len(images), len(labels)))

train_count = len(images)
train_data_set = []

# normalize data
pixel_type = numpy.dtype(numpy.uint8)

for i in range(train_count):
    image = numpy.frombuffer(images[i], dtype = pixel_type)
    image = (image - (numpy.iinfo(pixel_type).max - numpy.iinfo(pixel_type).min) / 2.0) / (numpy.iinfo(pixel_type).max - numpy.iinfo(pixel_type).min)

    train_data_set.append((image, labels[i]))

shuffle(train_data_set)

# create neural network and set up parameters
learning_rate = 0.001
network_layers = [image_height * image_width, 256, output_classes]

print "Creating neural network with layers {}".format(network_layers)
nn = neural_network(network_layers)

epochs = 256
batch_size = 64

plot = prediction_error_plot()

for epoch in range(epochs):

    processed_train_image = 0
    processed_train_image_error = 0
    processed_error = 0.0
    processed_batch = 0

    print 'Starting training epoch #{}'.format(epoch+1)
    print 'Batch size: {}'.format(batch_size)
    print 'Learning rate: {:.6f}'.format(learning_rate)
    print('')

    for i in range(train_count):

        image = train_data_set[i][0]
        label = train_data_set[i][1]
        output = nn.propagate_forward(image)

        correct = numpy.zeros((1, output_classes))
        correct[(0, label)] = 1.0

        # check prediction was correct
        predicted_class = 0
        predicted_value = 0
        for j in range(output_classes):
            if output[(0, j)] > predicted_value:
                predicted_value = output[(0, j)]
                predicted_class = j

        processed_train_image += 1

        # calculate error function
        error_delta = numpy.asarray(output - correct)
        error = error_delta / batch_size * learning_rate

        nn.propagate_backward(error)
        nn.update_delta()

        processed_error += numpy.sum(numpy.abs(error_delta * error_delta))

        if predicted_class != label:
            processed_train_image_error += 1

        # reset weights at the end of batch processing
        if processed_train_image and processed_train_image % batch_size == 0:

            nn.update_weights()
            print 'Weights delta: {}'.format(map (lambda x: float('%.4f' % x), nn.get_weights_delta()[0]))
            print 'Weights bias delta: {}'.format(map (lambda x: float('%.4f' % x), nn.get_weights_delta()[0]))
            print 'Batch total error: {:.4f}'.format(processed_error)
            nn.reset_weights_delta()

            processed_percentage_error = 100 * processed_train_image_error / processed_train_image
            print 'Prediction error: {:.0f}%'.format(100 * processed_train_image_error / processed_train_image)

            processed_train_image = 0
            processed_train_image_error = 0
            processed_error = 0.0
            processed_batch += 1

            if processed_batch and processed_batch % 16 == 0:
                plot.update(epoch + i * 1.0 / train_count, processed_percentage_error)

    print ''
    print 'Training epoch #{} was completed.'.format(epoch+1)
    print ''


print 'Training was completed successfully'
