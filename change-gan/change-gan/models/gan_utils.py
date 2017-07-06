""" Contains building blocks for various versions of GAN. """

import tensorflow as tf

slim = tf.contrib.slim


def leaky_relu_fn(negative_slope):
    def leaky_relu(x):
        return tf.maximum(negative_slope * x, x)
    return leaky_relu


def conv_layers(inputs, dims):
    inputs = slim.conv2d(inputs, dims[0], 4, stride=2,
                         activation_fn=leaky_relu_fn(0.2),
                         normalizer_fn=lambda x: x,
                         scope='Conv2d_0_{}'.format(dims[0]))

    for i, dim in enumerate(dims[1:]):
        inputs = slim.conv2d(inputs, dim, 4, stride=2,
                             activation_fn=leaky_relu_fn(0.2),
                             normalizer_fn=slim.batch_norm,
                             scope='Conv2d_{}_{}'.format(i + 1, dim))
    return inputs


def encoder(inputs, dims, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Encoder', [inputs], reuse=reuse):
        inputs = conv_layers(inputs, dims)
    return inputs


def decoder(inputs, dims, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Decoder', [inputs], reuse=reuse):
        for i, dim in enumerate(dims[:-1]):
            inputs = slim.conv2d_transpose(inputs, dim, 4, stride=2,
                                           normalizer_fn=slim.batch_norm,
                                           scope='Deconv2d_{}_{}'.format(i, dim))
        inputs = slim.conv2d_transpose(inputs, dims[-1], 4, stride=2,
                                       activation_fn=tf.tanh,
                                       normalizer_fn=lambda x: x,
                                       scope='Deconv2d_{}_{}'.format(len(dims) - 1, dims[-1]))
    return inputs


def discriminator(inputs, dims, scope=None, reuse=None, is_training=True):
    with tf.variable_scope(scope, 'Discriminator', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm],
                            is_training=is_training):
            inputs = conv_layers(inputs, dims)
            logits = slim.conv2d(inputs, 1, 4, stride=1, padding='VALID',
                                 activation_fn=lambda x: x,
                                 normalizer_fn=lambda x: x,
                                 scope='Logits'.format(len(dims)))
            probs = tf.nn.sigmoid(logits, name='Probs')
    return logits, probs


def preprocess_image(image, height, width, is_training=True):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, [height, width])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image
