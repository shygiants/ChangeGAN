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
                                is_training=True):
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

                bbox_channel_a = _get_bbox(inputs_a)
                bbox_channel_b = _get_bbox(inputs_b)

                outputs_ab, z_a_b = converter_ab(inputs_a)
                outputs_bbox_ab = tf.concat([outputs_ab, bbox_channel_a], 3)
                outputs_ba = converter_ba(inputs_b, z_a_b)
                outputs_bbox_ba = tf.concat([outputs_ba, bbox_channel_a], 3)

                outputs_bab, _ = converter_ab(outputs_bbox_ba, reuse=True)
                outputs_aba = converter_ba(outputs_bbox_ab, z_a_b, reuse=True)

                logits_a_real, probs_a_real = discriminator(inputs_a, deep_encoder_dims, scope='Discriminator_A')
                logits_a_fake, probs_a_fake = discriminator(outputs_bbox_ba, deep_encoder_dims, scope='Discriminator_A', reuse=True)
                logits_b_real, probs_b_real = discriminator(inputs_b, deep_encoder_dims, scope='Discriminator_B')
                logits_b_fake, probs_b_fake = discriminator(outputs_bbox_ab, deep_encoder_dims, scope='Discriminator_B', reuse=True)

                outputs = [_remove_bbox(inputs_a), _remove_bbox(inputs_b), outputs_ba, outputs_ab, outputs_aba, outputs_bab]
                outputs = map(lambda image: tf.image.convert_image_dtype(image, dtype=tf.uint8), outputs)

    with tf.name_scope('images'):
        tf.summary.image('X_A', _remove_bbox(inputs_a))
        tf.summary.image('X_B', _remove_bbox(inputs_b))
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
    l_const_a = tf.reduce_mean(tf.losses.absolute_difference(_remove_bbox(inputs_a), outputs_aba))
    l_const_b = tf.reduce_mean(tf.losses.absolute_difference(_remove_bbox(inputs_b), outputs_bab))
    l_color_a = tf.reduce_mean(tf.squared_difference(tf.reduce_mean(_remove_bbox(inputs_a) * bbox_channel_a, axis=[0, 1, 2]),
                                                     tf.reduce_mean(outputs_ab * bbox_channel_a, axis=[0, 1, 2])))
    l_color_b = tf.reduce_mean(tf.squared_difference(tf.reduce_mean(_remove_bbox(inputs_b) * bbox_channel_b, axis=[0, 1, 2]),
                                                     tf.reduce_mean(outputs_ba * bbox_channel_a, axis=[0, 1, 2])))

    l_g = l_g_a + l_g_b + 5. * (l_const_a + l_const_b) + 5. * (l_color_a + l_color_b)
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
        tf.summary.scalar('L_Color_A', l_color_a)
        tf.summary.scalar('L_Color_B', l_color_b)
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
    [image_a, bbox_a] = provider_a.get(['image', 'object/bbox'])
    [image_b, bbox_b] = provider_b.get(['image', 'object/bbox'])

    train_image_size = default_image_size

    def add_channel(image, bbox, padding='ZERO'):
        ymin = bbox[0]
        xmin = bbox[1]
        ymax = bbox[2]
        xmax = bbox[3]

        image_shape = tf.to_float(tf.shape(image))
        height = image_shape[0]
        width = image_shape[1]

        bbox_height = (ymax - ymin) * height
        bbox_width = (xmax - xmin) * width
        channel = tf.ones(tf.to_int32(tf.stack([bbox_height, bbox_width])))
        channel = tf.expand_dims(channel, axis=2)

        pad_top = tf.to_int32(ymin * height)
        pad_left = tf.to_int32(xmin * width)
        height = tf.to_int32(height)
        width = tf.to_int32(width)
        channel = tf.image.pad_to_bounding_box(channel, pad_top, pad_left, height, width)
        # TODO: Decide pad one or zero
        if padding == 'ONE':
            channel = tf.ones_like(channel) - channel

        image = tf.concat([image, channel], axis=2)

        return image

    image_a = tf.image.convert_image_dtype(image_a, dtype=tf.float32)
    image_b = tf.image.convert_image_dtype(image_b, dtype=tf.float32)
    # [Num of boxes, 4] => [4]
    bbox_a = tf.squeeze(bbox_a, axis=0)
    bbox_b = tf.squeeze(bbox_b, axis=0)
    # Add bound box as 4th channel
    image_a = add_channel(image_a, bbox_a)
    image_b = add_channel(image_b, bbox_b)

    image_space_a = Image(image_a, bbox_a)
    image_space_b = Image(image_b, bbox_b)

    # Resize image B
    ratio = image_space_a.bbox_height / image_space_b.bbox_height
    image_space_b.resize(ratio)

    # Shift image B to fit bboxes of two images
    pixel_shift = image_space_a.translate2pxl(image_space_a.bbox_center) - \
                  image_space_b.translate2pxl(image_space_b.bbox_center)

    # Calculate ymin and xmin
    crop_top = tf.less(pixel_shift[0], 0)
    pad_y = tf.cond(crop_top, true_fn=lambda: 0, false_fn=lambda: pixel_shift[0])
    crop_ymin = tf.cond(crop_top,
                        true_fn=lambda: image_space_b.translate2coor(pixel_y=tf.negative(pixel_shift[0])),
                        false_fn=lambda: 0.)
    crop_left = tf.less(pixel_shift[1], 0)
    pad_x = tf.cond(crop_left, true_fn=lambda: 0, false_fn=lambda: pixel_shift[1])
    crop_xmin = tf.cond(crop_left,
                        true_fn=lambda: image_space_b.translate2coor(pixel_x=tf.negative(pixel_shift[1])),
                        false_fn=lambda: 0.)

    # Calculate ymax and xmax
    over_y = pixel_shift[0] + image_space_b.height - image_space_a.height
    crop_bottom = tf.greater(over_y, 0)
    crop_ymax = tf.cond(crop_bottom,
                        true_fn=lambda: 1. - image_space_b.translate2coor(pixel_y=over_y),
                        false_fn=lambda: 1.)
    over_x = pixel_shift[1] + image_space_b.width - image_space_a.width
    crop_right = tf.greater(over_x, 0)
    crop_xmax = tf.cond(crop_right,
                        true_fn=lambda: 1. - image_space_b.translate2coor(pixel_x=over_x),
                        false_fn=lambda: 1.)

    # Resize, Crop, Pad
    image_b_cropped = image_space_b.crop(crop_ymin, crop_xmin, crop_ymax, crop_xmax)

    def pad_to_bounding_box(image):
        return tf.image.pad_to_bounding_box(image, pad_y, pad_x, image_space_a.height, image_space_a.width)

    # Pad differently depending on type of channel
    image_b_cropped, bbox_channel = _split_image_bbox(image_b_cropped)

    # One padding for RGB
    rgb_padding = pad_to_bounding_box(tf.ones_like(image_b_cropped))
    rgb_padding = tf.ones_like(rgb_padding) - rgb_padding
    # Sample background color and pad
    rgb_padding *= image_b_cropped[0, 0]

    # Pad for RGB
    image_b = pad_to_bounding_box(image_b_cropped) + rgb_padding

    # Zero padding for bbox channel
    bbox_channel = pad_to_bounding_box(bbox_channel)

    # Concat RGB and bbox channel
    image_b = tf.concat([image_b, bbox_channel], axis=2)

    # Preprocess images
    image_a = _preprocess_image(image_a, train_image_size, train_image_size, is_training=is_training)
    image_b = _preprocess_image(image_b, train_image_size, train_image_size, is_training=is_training)

    images_a, images_b, bboxes_a, bboxes_b = tf.train.batch(
        [image_a, image_b, bbox_a, bbox_b],
        batch_size=batch_size,
        num_threads=multiprocessing.cpu_count(),
        capacity=5 * batch_size)

    batch_queue = slim.prefetch_queue.prefetch_queue(
        [images_a, images_b, bboxes_a, bboxes_b], capacity=2)
    images_a, images_b, bboxes_a, bboxes_b = batch_queue.dequeue()

    with tf.name_scope('inputs'):
        tf.summary.image('X_A', _remove_bbox(images_a))
        tf.summary.image('X_A_BBox', images_a)
        tf.summary.image('X_B', _remove_bbox(images_b))
        tf.summary.image('X_B_BBox', images_b)

    return images_a, images_b, bboxes_a, bboxes_b


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


def _split_image_bbox(image_bbox):
    image, bbox = tf.split(image_bbox, [3, 1], axis=image_bbox.shape.ndims - 1)
    return image, bbox


def _remove_bbox(image_bbox):
    image, bbox = _split_image_bbox(image_bbox)
    return image


def _get_bbox(image_bbox):
    image, bbox = _split_image_bbox(image_bbox)
    return bbox


class Image:
    def __init__(self, image, bbox):
        self._image = image
        self._image_shape = tf.to_float(tf.shape(image))
        self._height = self._image_shape[0]
        self._width = self._image_shape[1]

        self._ratio = None

        self._bbox = bbox
        self._ymin = bbox[0]
        self._xmin = bbox[1]
        self._ymax = bbox[2]
        self._xmax = bbox[3]
        self._bbox_height = (self._ymax - self._ymin) * self._height
        self._bbox_width = (self._xmax - self._xmin) * self._width

        self._center_y = (self._ymin + self._ymax) / 2.
        self._center_x = (self._xmin + self._xmax) / 2.

    @property
    def image(self):
        return self._image

    @property
    def height(self):
        height = self._height
        if self._ratio is not None:
            height *= self._ratio
        return tf.to_int32(height)

    @property
    def width(self):
        width = self._width
        if self._ratio is not None:
            width *= self._ratio
        return tf.to_int32(width)

    @property
    def bbox_height(self):
        return self._bbox_height

    @property
    def bbox_center(self):
        return tf.stack([self._center_y, self._center_x])

    def resize(self, ratio):
        self._ratio = ratio

    def translate2pxl(self, coor):
        if coor.dtype != tf.float32:
            coor = tf.to_float(coor)
        pixel = coor * self._image_shape[:2]
        if self._ratio is not None:
            pixel *= self._ratio
        return tf.to_int32(pixel)

    def translate2coor(self, pixel_y=None, pixel_x=None):
        if pixel_y is None and pixel_x is None:
            raise ValueError
        if pixel_y is not None and pixel_x is not None:
            raise ValueError

        divisor = self._image_shape[0 if pixel_y is not None else 1]
        pixel = pixel_y if pixel_y is not None else pixel_x

        if pixel.dtype != tf.float32:
            pixel = tf.to_float(pixel)

        if self._ratio is not None:
            divisor *= self._ratio
        coor = pixel / divisor
        return coor

    def crop(self, ymin, xmin, ymax, xmax):
        image = self._image
        if self._ratio is not None:
            target_shape = tf.to_int32(self._image_shape[:2] * self._ratio)
            image = tf.image.resize_images(image, target_shape)

        shape = tf.to_float(tf.shape(image))
        height = shape[0]
        width = shape[1]

        offset_height = tf.to_int32(ymin * height)
        offset_width = tf.to_int32(xmin * width)
        target_height = tf.to_int32((ymax - ymin) * height)
        target_width = tf.to_int32((xmax - xmin) * width)
        image = tf.image.crop_to_bounding_box(image,
                                              offset_height,
                                              offset_width,
                                              target_height,
                                              target_width)

        return image
