""" Converts CelebA data to TFRecords file format with Example protos """

import sys
import os
import threading

import tensorflow as tf
import numpy as np

from datetime import datetime
from selector import get_image_files, check_attr_defined

tf.flags.DEFINE_string('image_dir', None,
                       'Images directory')
tf.flags.DEFINE_string('attributes_file', None,
                       'Attributes file path')
tf.flags.DEFINE_string('attributes', None, """
Images that have specified attributes are converted.

You may also pass a comma separated list of attributes, as in

python builder.py --attributes=blond_hair,black_hair
""")
tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')
tf.app.flags.DEFINE_integer('num_shards', 32,
                            'Number of shards in TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, height, width):
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/encoded': _bytes_feature(image_buffer)}))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

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


def _process_image(filename, coder):
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


def _process_image_files_batch(coder, thread_index, ranges, attribute, filenames, num_shards):
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (attribute, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]

            image_buffer, height, width = _process_image(filename, coder)

            example = _convert_to_example(filename, image_buffer, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(attribute, image_files, num_shards):
    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(image_files), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print 'Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges)
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, attribute, image_files, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(image_files)))
    sys.stdout.flush()


def _process_dataset(attribute, directory, num_shards):
    image_files = get_image_files(attribute, directory, FLAGS.attributes_file)
    _process_image_files(attribute, image_files, num_shards)


def main(_):
    assert not FLAGS.num_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.num_shards')
    assert FLAGS.attributes, (
        'FLAGS.attributes should be provided'
    )
    assert FLAGS.attributes_file, (
        'FLAGS.attributes_file should be provided'
    )
    assert FLAGS.image_dir, (
        'FLAGS.image_dir should be provided'
    )
    print 'Saving results to %s' % FLAGS.output_directory

    if not os.path.exists(FLAGS.output_directory):
        os.makedirs(FLAGS.output_directory)

    attributes = FLAGS.attributes.split(',')
    check_attr_defined(attributes)
    for attribute in attributes:
        _process_dataset(attribute, FLAGS.image_dir, FLAGS.num_shards)

if __name__ == '__main__':
    tf.app.run()
