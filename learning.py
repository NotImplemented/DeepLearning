import numpy
import neural_network
import mnist_database


test = neural_network.neural_network_test()
test.run_gradient_test()


database = mnist_database()
images = database.get_train_images()
labels = database.get_train_labels()

image_height = database.image_height
image_width = database.image_width
output_classes = 10

if len(images) != len(labels):
    raise ValueError('Train data set mismatch: {} labels, {} images'.format(len(images), len(labels)))

m = train_count = len(images)

nn = neural_network([image_height*image_width, 64, 32, output_classes])

processed_train_image = 0
processed_train_image_error = 0

for i in range(train_count):

    output = nn.propagate_forward(images[i])

    correct = numpy.zeros((1, output_classes))
    correct[(0, labels[i])] = 1.0

    # check prediction was correct
    max_predicted_class = 0
    max_predicted_value = 0
    for j in range(output_classes):
        if output[(0, j)] > max_predicted_value:
            max_predicted_value = output[(0, j)]
            max_predicted_class = j

    processed_train_image += 1

    if max_predicted_class != labels[i]:
        processed_train_image_error += 1

    if processed_train_image and processed_train_image % 128 == 0:
        processed_percentage_error = 100 * processed_train_image_error / processed_train_image
        print 'Processed: {} images, learning error: {.4f} %'.format(processed_train_image, processed_percentage_error)

    # calculate error function
    error_delta = output - correct
    error = (error_delta * error_delta) / m

    # propagate and update weights
    nn.propagate_backward(error)
    nn.update_weights()

print 'Training was completed successfully'
print 'Processed: {} images, learning error: {.4f} %'.format(processed_train_image, processed_percentage_error)


