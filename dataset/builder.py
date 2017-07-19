""" Converts CelebA data to TFRecords file format with Example protos """

from __future__ import division

import sys
import os
import threading
import itertools
import random

import tensorflow as tf
import numpy as np

from datetime import datetime
from selectors import deepfashion, crawled, celeba
import utils


tf.flags.DEFINE_string('attributes', None, """
Images that have specified attributes are converted.

You may also pass a comma separated list of attributes, as in

python builder.py --attributes=blond_hair,black_hair
""")
tf.app.flags.DEFINE_boolean('bound_box', False,
                            'Whether to encode bound boxes.')
tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')
tf.app.flags.DEFINE_integer('num_shards', 32,
                            'Number of shards in TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_integer('num_images', None,
                            'Number of images to convert.')

FLAGS = tf.app.flags.FLAGS


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

        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)

        _write_examples(writer, coder, filenames, files_in_shard)

        writer.close()
        counter += len(files_in_shard)
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, len(files_in_shard), output_file))
        sys.stdout.flush()
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
    coder = utils.ImageCoder()

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


def _process_dataset(attribute, num_shards):
    image_files = []
    if attribute in celeba._PREDEFINED_ATTR:
        image_files = celeba.get_image_files(attribute)
    elif attribute == 'clothes':
        image_files.append(deepfashion.get_image_files('top,flat'))
        image_files.append(crawled.get_image_files('clothes'))
    elif attribute == 'models':
        image_files.append(deepfashion.get_image_files('top,front'))
        image_files.append(crawled.get_image_files('models'))
    else:
        raise LookupError
    image_files = list(itertools.chain.from_iterable(image_files))

    if FLAGS.num_images is not None:
        image_files = image_files[:FLAGS.num_images]

    _process_image_files(attribute, image_files, num_shards)


def _process_bbox_dataset(split_ratio=0.9):
    image_files = []
    bboxes = []
    labels = []

    images_top, bboxes_top = deepfashion.get_image_with_bbox('top,front')
    image_files.append(images_top)
    bboxes.append(bboxes_top)
    labels.append([1] * len(bboxes_top))

    images_bottom, bboxes_bottom = deepfashion.get_image_with_bbox('bottom,front')
    image_files.append(images_bottom)
    bboxes.append(bboxes_bottom)
    labels.append([2] * len(bboxes_bottom))

    image_files = list(itertools.chain.from_iterable(image_files))
    bboxes = list(itertools.chain.from_iterable(bboxes))
    labels = list(itertools.chain.from_iterable(labels))

    assert len(image_files) == len(bboxes)
    assert len(labels) == len(bboxes)
    num_items = len(image_files)
    indices = range(num_items)
    random.shuffle(indices)

    eval_start = int(num_items * split_ratio)

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = utils.ImageCoder()

    # Write train file
    train_file = os.path.join(FLAGS.output_directory, 'train_clothes.tfrecords')
    train_writer = tf.python_io.TFRecordWriter(train_file)
    _write_examples(train_writer, coder, image_files, indices[:eval_start], bboxes=bboxes, labels=labels)

    # Write eval file
    eval_file = os.path.join(FLAGS.output_directory, 'eval_clothes.tfrecords')
    eval_writer = tf.python_io.TFRecordWriter(eval_file)
    _write_examples(eval_writer, coder, image_files, indices[eval_start:], bboxes=bboxes, labels=labels)


def _write_examples(writer, coder, filenames, indices, bboxes=None, labels=None):
    for i in indices:
        filename = filenames[i]

        image_buffer, height, width = utils.process_image(filename, coder)

        if bboxes is not None and labels is not None:
            bbox = bboxes[i]
            label = labels[i]
            example = utils.convert_to_example(filename, image_buffer, height, width, bbox, label)
        else:
            example = utils.convert_to_example(filename, image_buffer, height, width)
        writer.write(example.SerializeToString())


def main(_):
    assert not FLAGS.num_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.num_shards')
    assert FLAGS.bound_box or FLAGS.attributes, (
        'FLAGS.attributes should be provided'
    )
    print 'Saving results to %s' % FLAGS.output_directory

    if not os.path.exists(FLAGS.output_directory):
        os.makedirs(FLAGS.output_directory)
    if FLAGS.bound_box:
        _process_bbox_dataset()
        return
    attributes = FLAGS.attributes.split(',')
    for attribute in attributes:
        _process_dataset(attribute, FLAGS.num_shards)

if __name__ == '__main__':
    tf.app.run()
