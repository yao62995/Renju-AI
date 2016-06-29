#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com>

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("threads", 5, "number of thread to run")


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_images(images, labels, writer):
    """ Construct distorted input for Image training using the Reader ops.
    :param images: Images. 4D tensor of [batch_size, IMAGE_Height, IMAGE_Width, Channel] size.
    :param labels: Labels. 1D tensor of [batch_size] size.
    :param writer:  an instance of TFRecordWriter
    :return: batch_size
    """
    assert len(images) == len(labels), "Number of images, %d should be equal to number of labels %d" % \
                                       (len(images), len(labels))
    return serialize_data(images, labels, writer)


def serialize_data(images, labels, writer):
    """
    :param images: Images. 4D tensor of [batch_size, IMAGE_Height, IMAGE_Width, Channel] size.
    :param labels: Labels. 1D tensor of [batch_size] size.
    :param writer: an instance of TFRecordWriter
    :return: num_examples
    """
    num_examples = len(images)
    assert num_examples == len(labels), "Number of images, %d should be equal to number of labels %d" % \
                                        (num_examples, len(labels))
    if num_examples == 0:
        return num_examples
    height, width, channel = images[0].shape
    for index in xrange(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': int64_feature(height),
            'width': int64_feature(width),
            'channel': int64_feature(channel),
            'label': int64_feature(int(labels[index])),
            'image_raw': bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    return num_examples


def read_and_decode(filename_queue, shape, normalize=False, flatten=True):
    """Reads
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            # 'height': tf.FixedLenFeature([], tf.int64),
            # 'width': tf.FixedLenFeature([], tf.int64),
            # 'channel': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # height, width, channel = features['height'], features['width'], features['channel']
    height, width, channel = shape

    if flatten:
        num_elements = height * width * channel
        image = tf.reshape(image, [num_elements])
        image.set_shape(num_elements)
    else:
        image = tf.reshape(image, (width, height, channel))
        image.set_shape((width, height, channel))

    if normalize:
        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32)
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label


def distorted_inputs(file_records, shape, batch_size=64,
                     num_epochs=None, num_threads=5, num_examples_per_epoch=128,
                     flatten=False, normalize=False):
    """Construct distorted input for training using the Reader ops.
    :param file_records:  a list of tf_records files
    :param num_epochs: an integer, if specified, string_input_producer` produces each string from
                                `string_tensor` `num_epochs` times before generating an `OutOfRange` error.
                                If not specified, `string_input_producer` can cycle an unlimited number of times.
    """
    if type(file_records) is str:
        file_records = [file_records]

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            file_records, num_epochs=num_epochs, name='string_DISTORTED_input_producer')

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = read_and_decode(filename_queue, shape=shape, flatten=flatten, normalize=normalize)

        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # Removed random_crop in new TensorFlow release.
        # Randomly crop a [height, width] section of the image.
        # distorted_image = tf.image.random_crop(image, [height, width])
        #
        # Randomly flip the image horizontally.
        # distorted_image = tf.image.random_flip_left_right(image)
        #
        # Because these operations are not commutative, consider randomizing
        # randomize the order their operation.
        # distorted_image = tf.image.random_brightness(distorted_image,
        #                                              max_delta=63)
        # distorted_image = tf.image.random_contrast(distorted_image,
        #                                            lower=0.2, upper=1.8)

        # # Subtract off the mean and divide by the variance of the pixels.
        # float_image = tf.image.per_image_whitening(distorted_image)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)
        images, sparse_labels = tf.train.shuffle_batch([image, label],
                                                       batch_size=batch_size,
                                                       num_threads=num_threads,
                                                       capacity=min_queue_examples + 3 * batch_size,
                                                       enqueue_many=False,
                                                       # Ensures a minimum amount of shuffling of examples.
                                                       min_after_dequeue=min_queue_examples,
                                                       name='batching_shuffling_distortion')

    return images, sparse_labels
