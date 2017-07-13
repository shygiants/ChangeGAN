""" Contains the definition of the ChangeGAN architecture. """
import multiprocessing

import tensorflow as tf

from gan_utils import encoder, decoder, transformer, discriminator, preprocess_image

slim = tf.contrib.slim

default_image_size = 256


def model_fn(inputs_a, inputs_b, learning_rate, num_blocks=9, is_training=True, scope=None, weight_decay=0.0001):
    encoder_dims = [32, 64, 128]
    deep_encoder_dims = [64, 128, 256]
    decoder_dims = [64, 32, 3]
    deep_decoder_dims = [128, 64, 3]
    with tf.variable_scope(scope, 'ChangeGAN', [inputs_a, inputs_b]):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.batch_norm],
                                is_training=is_training):
                def converter_ab(inputs_a, reuse=None):
                    ################
                    # Encoder part #
                    ################
                    z_a = encoder(inputs_a, deep_encoder_dims, scope='Encoder_A', reuse=reuse)

                    # z_a is split into c_b, z_a-b
                    c_b, z_a_b = tf.split(z_a, num_or_size_splits=2, axis=3)

                    ####################
                    # Transformer part #
                    ####################
                    c_b = transformer(c_b, encoder_dims[-1], num_blocks=num_blocks,
                                      scope='Transformer_B', reuse=reuse)

                    ################
                    # Decoder part #
                    ################
                    outputs_b = decoder(c_b, decoder_dims, scope='Decoder_B', reuse=reuse)

                    return outputs_b, z_a_b

                def converter_ba(inputs_b, z_a_b, reuse=None):
                    ################
                    # Encoder part #
                    ################
                    z_b = encoder(inputs_b, encoder_dims, scope='Encoder_B', reuse=reuse)

                    # Concat z_b and z_a-b
                    c_a = tf.concat([z_b, z_a_b], 3)

                    ####################
                    # Transformer part #
                    ####################
                    c_a = transformer(c_a, deep_encoder_dims[-1], num_blocks=num_blocks,
                                      scope='Transformer_A', reuse=reuse)

                    ################
                    # Decoder part #
                    ################
                    outputs_a = decoder(c_a, deep_decoder_dims, scope='Decoder_A', reuse=reuse)

                    return outputs_a

                outputs_ab, z_a_b = converter_ab(inputs_a)
                outputs_ba = converter_ba(inputs_b, z_a_b)
                outputs_bab, _ = converter_ab(outputs_ba, reuse=True)
                outputs_aba = converter_ba(outputs_ab, z_a_b, reuse=True)

                logits_a_real, probs_a_real = discriminator(inputs_a, deep_encoder_dims, scope='Discriminator_A')
                logits_a_fake, probs_a_fake = discriminator(outputs_ba, deep_encoder_dims, scope='Discriminator_A', reuse=True)
                logits_b_real, probs_b_real = discriminator(inputs_b, deep_encoder_dims, scope='Discriminator_B')
                logits_b_fake, probs_b_fake = discriminator(outputs_ab, deep_encoder_dims, scope='Discriminator_B', reuse=True)

                outputs = [outputs_ba, outputs_ab, outputs_aba, outputs_bab]

    with tf.name_scope('images'):
        tf.summary.image('X_A', inputs_a)
        tf.summary.image('X_B', inputs_b)
        tf.summary.image('X_BA', outputs_ba)
        tf.summary.image('X_AB', outputs_ab)
        tf.summary.image('X_ABA', outputs_aba)
        tf.summary.image('X_BAB', outputs_bab)

    global_step = tf.train.get_or_create_global_step()

    if not is_training:
        return outputs

    t_vars = tf.trainable_variables()

    d_a_vars = [var for var in t_vars if 'Discriminator_A' in var.name]
    d_b_vars = [var for var in t_vars if 'Discriminator_B' in var.name]
    g_vars = [var for var in t_vars if 'coder' in var.name or 'Transformer' in var.name]

    ##########
    # Losses #
    ##########
    # Losses for discriminator
    l_d_a_fake = tf.reduce_mean(tf.square(logits_a_fake))
    l_d_a_real = tf.reduce_mean(tf.squared_difference(logits_a_real, 1.))
    l_d_a = 0.5 * (l_d_a_fake + l_d_a_real)
    train_op_d_a = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.5,
        beta2=0.999
    ).minimize(l_d_a, global_step=global_step, var_list=d_a_vars)

    l_d_b_fake = tf.reduce_mean(tf.square(logits_b_fake))
    l_d_b_real = tf.reduce_mean(tf.squared_difference(logits_b_real, 1.))
    l_d_b = 0.5 * (l_d_b_fake + l_d_b_real)
    train_op_d_b = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.5,
        beta2=0.999
    ).minimize(l_d_b, global_step=global_step, var_list=d_b_vars)

    l_d = l_d_a + l_d_b

    # Losses for generators
    l_g_a = tf.reduce_mean(tf.squared_difference(logits_a_fake, 1.))
    l_g_b = tf.reduce_mean(tf.squared_difference(logits_b_fake, 1.))
    l_const_a = tf.reduce_mean(tf.losses.absolute_difference(inputs_a, outputs_aba))
    l_const_b = tf.reduce_mean(tf.losses.absolute_difference(inputs_b, outputs_bab))

    l_g = l_g_a + l_g_b + 10. * (l_const_a + l_const_b)
    train_op_g = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.5,
        beta2=0.999
    ).minimize(l_g, global_step=global_step, var_list=g_vars)

    with tf.name_scope('losses'):
        tf.summary.scalar('L_D_A_Fake', l_d_a_fake)
        tf.summary.scalar('L_D_A_Real', l_d_a_real)
        tf.summary.scalar('L_D_A', l_d_a)
        tf.summary.scalar('L_D_B_Fake', l_d_b_fake)
        tf.summary.scalar('L_D_B_Real', l_d_b_real)
        tf.summary.scalar('L_D_B', l_d_b)
        tf.summary.scalar('L_D', l_d)

        tf.summary.scalar('L_G_A', l_g_a)
        tf.summary.scalar('L_G_B', l_g_b)
        tf.summary.scalar('L_Const_A', l_const_a)
        tf.summary.scalar('L_Const_B', l_const_b)
        tf.summary.scalar('L_G', l_g)

    train_op = tf.group(*[train_op_d_a, train_op_d_b, train_op_g])

    return train_op, global_step, outputs


def input_fn(dataset_a, dataset_b, batch_size=1, num_readers=4, is_training=True):
    provider_a = slim.dataset_data_provider.DatasetDataProvider(
        dataset_a,
        num_readers=num_readers,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)
    provider_b = slim.dataset_data_provider.DatasetDataProvider(
        dataset_b,
        num_readers=num_readers,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)
    [image_a] = provider_a.get(['image'])
    [image_b] = provider_b.get(['image'])

    train_image_size = default_image_size

    image_a = _preprocess_image(image_a, train_image_size, train_image_size, is_training=is_training)
    image_b = _preprocess_image(image_b, train_image_size, train_image_size, is_training=is_training)

    images_a, images_b = tf.train.batch(
        [image_a, image_b],
        batch_size=batch_size,
        num_threads=multiprocessing.cpu_count(),
        capacity=5 * batch_size)

    batch_queue = slim.prefetch_queue.prefetch_queue(
        [images_a, images_b], capacity=2)
    images_a, images_b = batch_queue.dequeue()

    return images_a, images_b


def _preprocess_image(image, height, width, is_training=True):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Central square crop and resize
    shape = tf.to_float(tf.shape(image))
    original_height = shape[0]
    original_width = shape[1]
    rest = (1. - original_width / original_height) / 2.
    image = tf.expand_dims(image, 0)
    images = tf.image.crop_and_resize(image,
                                      [[rest, 0., 1. - rest, 1.]], [0],
                                      [height, width])
    image = tf.squeeze(images, [0])
    # image = tf.image.resize_images(image, [height, width])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    return image
