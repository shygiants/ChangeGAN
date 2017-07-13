""" Contains building blocks for various versions of GAN. """

import tensorflow as tf

slim = tf.contrib.slim


def reflection_pad(inputs, padding):
    pad_2d = [padding] * 2
    inputs = tf.pad(inputs, [[0, 0], pad_2d, pad_2d, [0, 0]], mode='REFLECT')
    return inputs


def leaky_relu_fn(negative_slope):
    def leaky_relu(x):
        return tf.maximum(negative_slope * x, x)
    return leaky_relu


def conv_layers(inputs, dims):
    inputs = reflection_pad(inputs, 3)
    inputs = slim.conv2d(inputs, dims[0], 7,
                         normalizer_fn=slim.batch_norm,
                         padding='VALID',
                         scope='Conv2d_0_{}'.format(dims[0]))

    for i, dim in enumerate(dims[1:]):
        inputs = slim.conv2d(inputs, dim, 3, stride=2,
                             normalizer_fn=slim.batch_norm,
                             scope='Conv2d_{}_{}'.format(i + 1, dim))
    return inputs


def resize_deconv(inputs, num_features, kernel_size, scope='Resize_Deconv'):
    shape = tf.shape(inputs)
    inputs = tf.image.resize_images(inputs, [2 * shape[1], 2 * shape[2]],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    inputs = reflection_pad(inputs, 1)
    inputs = slim.conv2d(inputs, num_features, kernel_size,
                         normalizer_fn=slim.batch_norm,
                         padding='VALID', scope=scope)

    return inputs


def resnet_block(inputs, num_features, scope=None, reuse=None):
    shortcut = inputs
    with tf.variable_scope(scope, 'ResNet_Block', [inputs], reuse=reuse):
        inputs = reflection_pad(inputs, 1)
        inputs = slim.conv2d(inputs, num_features, 3,
                             normalizer_fn=slim.batch_norm, padding='VALID',
                             scope='Conv2d_0_{}'.format(num_features))
        inputs = reflection_pad(inputs, 1)
        inputs = slim.conv2d(inputs, num_features, 3, activation_fn=lambda x: x,
                             normalizer_fn=slim.batch_norm, padding='VALID',
                             scope='Conv2d_1_{}'.format(num_features))
    return inputs + shortcut


def encoder(inputs, dims, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Encoder', [inputs], reuse=reuse):
        inputs = conv_layers(inputs, dims)
    return inputs


def decoder(inputs, dims, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Decoder', [inputs], reuse=reuse):
        for i, dim in enumerate(dims[:-1]):
            inputs = resize_deconv(inputs, dim, 3,
                                   scope='Deconv2d_{}_{}'.format(i, dim))

        inputs = reflection_pad(inputs, 3)
        inputs = slim.conv2d(inputs, dims[-1], 7,
                             normalizer_fn=lambda x: x,
                             activation_fn=tf.tanh,
                             padding='VALID',
                             scope='Deconv2d_{}_{}'.format(len(dims) - 1, dims[-1]))
    return inputs


def transformer(inputs, num_features, num_blocks=6, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Transformer', [inputs], reuse=reuse):
        for i in range(num_blocks):
            inputs = resnet_block(inputs, num_features, scope='ResNet_Block_{}'.format(i))
    return inputs


def discriminator(inputs, dims, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Discriminator', [inputs], reuse=reuse):
        inputs = slim.conv2d(inputs, dims[0], 4, stride=2,
                             activation_fn=leaky_relu_fn(0.2),
                             normalizer_fn=lambda x: x,
                             scope='Conv2d_0_{}'.format(dims[0]))

        for i, dim in enumerate(dims[1:]):
            inputs = slim.conv2d(inputs, dim, 4, stride=2,
                                 activation_fn=leaky_relu_fn(0.2),
                                 normalizer_fn=slim.batch_norm,
                                 scope='Conv2d_{}_{}'.format(i + 1, dim))

        inputs = slim.conv2d(inputs, dims[-1], 4,
                             activation_fn=leaky_relu_fn(0.2),
                             normalizer_fn=slim.batch_norm,
                             scope='Conv2d_{}_{}'.format(len(dims), dims[-1]))

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
