import urllib
import struct
import numpy
import os.path
import gzip


class mnist_database:

    test_file_images = 't10k-images-idx3-ubyte'
    test_file_labels = 't10k-labels-idx1-ubyte'

    train_file_images = 'train-images-idx3-ubyte'
    train_file_labels = 'train-labels-idx1-ubyte'

    image_height = -1
    image_width = -1

    database_location = 'http://yann.lecun.com/exdb/mnist/'

    def __init__(self):
        self.test_file_images_data = None
        self.test_file_labels_data = None

        self.train_file_images_data = None
        self.train_file_labels_data = None


    def get_file(self, filename):

        if not os.path.exists(filename + '.gz'):

            url = self.database_location + filename + '.gz'
            print('Getting {} ...'.format(url))

            urllib.urlretrieve(self.database_location + filename + '.gz', filename + '.gz')
            print('Done')


    def read_images(self, filename):
        with gzip.open(filename + '.gz', 'rb') as file:
            magic, = struct.unpack('>i', file.read(4))

            if magic != 2051:
                raise ValueError("Magic number {} differs from {} in '{}'".format(magic, 2051, filename))

            count, = struct.unpack('>i', file.read(4))

            self.image_height, = rows, = struct.unpack('>i', file.read(4))
            self.image_width, = columns, = struct.unpack('>i', file.read(4))

            images = []

            for i in range(count):
                images.append(file.read(rows * columns))

            return images

    def read_labels(self, filename):
        with gzip.open(filename + '.gz', 'rb') as file:
            magic, = struct.unpack('>i', file.read(4))

            if magic != 2049:
                raise ValueError("Magic number {} differs from {} in '{}'".format(magic, 2049, filename))

            count, = struct.unpack('>i', file.read(4))
            return file.read(count)

    def get_test_images(self):

        if self.test_file_images_data:
            return self.test_file_images_data

        self.get_file(self.test_file_images)
        self.test_file_images_data = self.read_images(self.test_file_images)

        return self.test_file_images_data

    def get_test_labels(self):

        if self.test_file_labels_data:
            return self.test_file_labels_data

        self.get_file(self.test_file_labels)
        self.test_file_labels_data = self.read_labels(self.test_file_labels)

        return self.test_file_labels_data

    def get_train_images(self):

        if self.train_file_images_data:
            return self.train_file_images_data

        self.get_file(self.train_file_images)
        self.train_file_images_data = self.read_images(self.train_file_images)

        return self.train_file_images_data

    def get_train_labels(self):

        if self.train_file_labels_data:
            return self.train_file_labels_data

        self.get_file(self.train_file_labels)
        self.train_file_labels_data = self.read_labels(self.train_file_labels)

        return self.train_file_labels_data

    def load_data(self):
        self.get_test_images()
        self.get_test_labels()

        self.get_train_images()
        self.get_train_labels()