import numpy
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


pixel_min = 0
pixel_max = 255

for i in range(train_count):
    image = numpy.frombuffer(images[i], dtype = numpy.dtype(numpy.uint8))
    image = (image - (pixel_max - pixel_min)) / pixel_max

    train_data_set.append((image, labels[i]))

shuffle(train_data_set)

learning_rate = 30.0
network_layers = [image_height * image_width, 64, 32, output_classes]
print "Creating neural network with layers {}".format(network_layers)
nn = neural_network([image_height * image_width, 64, 32, output_classes])

epochs = 256
batch_size = 1200

for e in range(epochs):

    processed_train_image = 0
    processed_train_image_error = 0
    print 'Starting training epoch #{}'.format(e+1)
    print('')

    for i in range(train_count):

        image = train_data_set[i][0]
        label = train_data_set[i][1]
        output = nn.propagate_forward(image)

        correct = numpy.zeros((1, output_classes))
        correct[(0, label)] = 1.0

        # check prediction was correct
        max_predicted_class = 0
        max_predicted_value = 0
        for j in range(output_classes):
            if output[(0, j)] > max_predicted_value:
                max_predicted_value = output[(0, j)]
                max_predicted_class = j

        processed_train_image += 1

        # calculate error function
        error_delta = numpy.asarray(output - correct)
        error = (error_delta * error_delta) / batch_size * learning_rate

        nn.propagate_backward(error)
        nn.update_delta()

        if max_predicted_class != labels[i]:
            processed_train_image_error += 1

        # print learning error
        if processed_train_image and processed_train_image % batch_size == 0:

            nn.update_weights()
            print 'Weights delta: {}'.format(nn.get_weights_delta()[0])
            print 'Weights bias delta: {}'.format(nn.get_weights_delta()[1])
            nn.reset_weights_delta()

            processed_percentage_error = 100 * processed_train_image_error / processed_train_image
            print 'Processed: {} images, learning error: {:.2f} %'.format(processed_train_image, processed_percentage_error)

            processed_train_image = 0
            processed_train_image_error = 0

    print ''
    print 'Training epoch #{} was completed. Processed: {} images, learning error: {:.2f} %'.format(e+1, processed_train_image, processed_percentage_error)
    print ''


print 'Training was completed successfully'
