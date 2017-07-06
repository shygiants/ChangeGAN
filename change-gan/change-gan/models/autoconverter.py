""" Contains the definition of the Autoconverter architecture. """
import multiprocessing

import tensorflow as tf

from gan_utils import encoder, decoder, discriminator, preprocess_image

slim = tf.contrib.slim

default_image_size = 128


def model_fn(inputs_a, inputs_b, learning_rate, is_training=True, scope=None):
    encoder_dims = [64, 128, 256, 512]
    decoder_dims = [256, 128, 64, 3]
    with tf.variable_scope(scope, 'Autoconverter', [inputs_a, inputs_b]):
        with slim.arg_scope([slim.batch_norm],
                            is_training=is_training):
            def autoencoder(inputs_a, inputs_b, reuse=None):
                z_a = encoder(inputs_a, encoder_dims, scope='Encoder_A', reuse=reuse)
                z_b = encoder(inputs_b, encoder_dims, scope='Encoder_B', reuse=reuse)
                outputs_a_a = decoder(z_a, decoder_dims, scope='Decoder_A', reuse=reuse)
                outputs_b_b = decoder(z_b, decoder_dims, scope='Decoder_B', reuse=reuse)
                return outputs_a_a, outputs_b_b

            def converter(inputs_a, inputs_b, reuse=None):
                z_a = encoder(inputs_a, encoder_dims, scope='Encoder_A', reuse=reuse)
                z_b = encoder(inputs_b, encoder_dims, scope='Encoder_B', reuse=reuse)
                outputs_a_b = decoder(z_a, decoder_dims, scope='Decoder_B', reuse=reuse)
                outputs_b_a = decoder(z_b, decoder_dims, scope='Decoder_A', reuse=reuse)
                return outputs_a_b, outputs_b_a

            ####################
            # Autoencoder part #
            ####################
            outputs_a_a, outputs_b_b = autoencoder(inputs_a, inputs_b)

            # TODO: Decide whether the outputs of autoencoder are fed to discriminator
            # discriminator(outputs_a_a, encoder_dims, scope='Discriminator_A', reuse=True)
            # discriminator(outputs_b_b, encoder_dims, scope='Discriminator_B', reuse=True)

            #############################
            # Converter (DiscoGAN) part #
            #############################
            outputs_a_b, outputs_b_a = converter(inputs_a, inputs_b, reuse=True)

            logits_a_fake, probs_a_fake = discriminator(outputs_b_a, encoder_dims, scope='Discriminator_A')
            logits_a_real, probs_a_real = discriminator(inputs_a, encoder_dims, scope='Discriminator_A', reuse=True)

            logits_b_fake, probs_b_fake = discriminator(outputs_a_b, encoder_dims, scope='Discriminator_B')
            logits_b_real, probs_b_real = discriminator(inputs_b, encoder_dims, scope='Discriminator_B', reuse=True)

            outputs_b_a_b, outputs_a_b_a = converter(outputs_b_a, outputs_a_b, reuse=True)

            outputs = [outputs_a_a, outputs_b_b, outputs_a_b, outputs_b_a, outputs_a_b_a, outputs_b_a_b]

    with tf.name_scope('images'):
        tf.summary.image('X_A', inputs_a)
        tf.summary.image('X_B', inputs_b)
        tf.summary.image('X_BA', outputs_b_a)
        tf.summary.image('X_AB', outputs_a_b)
        tf.summary.image('X_AA', outputs_a_a)
        tf.summary.image('X_BB', outputs_b_b)
        tf.summary.image('X_ABA', outputs_a_b_a)
        tf.summary.image('X_BAB', outputs_b_a_b)

    global_step = tf.train.get_or_create_global_step()

    if not is_training:
        return outputs

    t_vars = tf.trainable_variables()

    d_vars = [var for var in t_vars if 'Discriminator' in var.name]
    g_vars = [var for var in t_vars if 'coder' in var.name]

    ##########
    # Losses #
    ##########
    # Losses for discriminator
    l_d_a_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits_a_fake), logits_a_fake))
    l_d_a_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_a_real), logits_a_real))
    l_d_a = l_d_a_fake + l_d_a_real

    l_d_b_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.zeros_like(logits_b_fake), logits_b_fake))
    l_d_b_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_b_real), logits_b_real))
    l_d_b = l_d_b_fake + l_d_b_real

    l_d = l_d_a + l_d_b
    train_op_d = tf.train.AdamOptimizer(
        learning_rate=learning_rate
    ).minimize(l_d, global_step=global_step, var_list=d_vars)

    # Losses for discriminator
    l_g_a = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_a_fake), logits_a_fake))
    l_g_b = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(logits_b_fake), logits_b_fake))

    # Autoencoder reconstruction
    l_const_a_auto = tf.reduce_mean(tf.losses.mean_squared_error(inputs_a, outputs_a_a))
    l_const_b_auto = tf.reduce_mean(tf.losses.mean_squared_error(inputs_b, outputs_b_b))

    # Converter reconstruction
    l_const_a_conv = tf.reduce_mean(tf.losses.mean_squared_error(inputs_a, outputs_a_b_a))
    l_const_b_conv = tf.reduce_mean(tf.losses.mean_squared_error(inputs_b, outputs_b_a_b))

    l_g = l_g_a + l_g_b + l_const_a_auto + l_const_b_auto + l_const_a_conv + l_const_b_conv
    train_op_g = tf.train.AdamOptimizer(
        learning_rate=learning_rate
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
        tf.summary.scalar('L_Const_A_Auto', l_const_a_auto)
        tf.summary.scalar('L_Const_B_Auto', l_const_b_auto)
        tf.summary.scalar('L_Const_A_Conv', l_const_a_conv)
        tf.summary.scalar('L_Const_B_Conv', l_const_b_conv)
        tf.summary.scalar('L_G', l_g)

    train_op = tf.group(*[train_op_d, train_op_g])

    return train_op, global_step, outputs


def input_fn(dataset_a, dataset_b, batch_size=32, num_readers=4, is_training=True):
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

    image_a = preprocess_image(image_a, train_image_size, train_image_size, is_training=is_training)
    image_b = preprocess_image(image_b, train_image_size, train_image_size, is_training=is_training)

    images_a, images_b = tf.train.batch(
        [image_a, image_b],
        batch_size=batch_size,
        num_threads=multiprocessing.cpu_count(),
        capacity=5 * batch_size)

    return images_a, images_b
