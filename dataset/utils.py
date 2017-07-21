""" Dataset Utils """

from __future__ import division

import os

import tensorflow as tf
import numpy as np


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

_classes_text = ['Top', 'Bottom']


def convert_to_example(filename, image_buffer, height, width, bbox=None, label=None, bbox_normalize=False):
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    if bbox is None:
        example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': int64_feature(height),
          'image/width': int64_feature(width),
          'image/colorspace': bytes_feature(colorspace),
          'image/channels': int64_feature(channels),
          'image/format': bytes_feature(image_format),
          'image/filename': bytes_feature(os.path.basename(filename)),
          'image/encoded': bytes_feature(image_buffer)}))
    else:
        image_format = b'jpg'

        ymin, xmin, ymax, xmax = bbox
        if not bbox_normalize:
            xmin /= width
            xmax /= width
            ymin /= height
            ymax /= height

        label_text = _classes_text[label - 1]

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
            'image/filename': bytes_feature(os.path.basename(filename)),
            'image/source_id': bytes_feature(os.path.basename(filename)),
            'image/encoded': bytes_feature(image_buffer),
            'image/format': bytes_feature(image_format),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/class/text': bytes_feature(label_text),
            'image/object/class/label': int64_feature(label),
        }))

    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self, ssd_graph_path=None):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        if ssd_graph_path is not None:
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(ssd_graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            # TODO: Label
            # label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
            # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
            #                                                             use_display_name=True)
            # category_index = label_map_util.create_category_index(categories)

            detection_graph = self._sess.graph
            self._encoded_image = detection_graph.get_tensor_by_name('encoded_image_string_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            self._detect_objects = [boxes, scores, classes, num_detections]

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb,
                              feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def detect_objects(self, image_data):
        outputs = self._sess.run(
            self._detect_objects, feed_dict={self._encoded_image: image_data})
        outputs = map(np.squeeze, outputs[:3])

        # TODO: General
        # Filter only tops
        classes = outputs[2]
        indices = classes == 1
        outputs = map(lambda l: l[indices], outputs)
        [box, score, class_] = map(lambda l: l[0], outputs)

        return box, score, class_


def process_image(filename, coder):
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()

    # # Clean the dirty data.
    # if _is_cmyk(filename):
    #     # 22 JPEG images are in CMYK colorspace.
    #     print('Converting CMYK to RGB for %s' % filename)
    #     image_data = coder.cmyk_to_rgb(image_data)

    try:
        # Decode the RGB JPEG.
        image = coder.decode_jpeg(image_data)
    except tf.errors.InvalidArgumentError:
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def detect_objects(image_data, coder):
    return coder.detect_objects(image_data)
