
import os
import tensorflow as tf

slim = tf.contrib.slim

_FILE_PATTERN = '%s-*'

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'object/bbox': 'A list of bounding boxes.',
}

_DOMAINS_TO_SIZES = {
    'clothes': 13652,
    'models': 11243,
}


def get_dataset(domain_name, dataset_dir, file_pattern=None, reader=None):
    if domain_name not in _DOMAINS_TO_SIZES:
        raise ValueError('domain name %s was not recognized.' % domain_name)
    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % domain_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature(
            (), tf.string, default_value='jpeg'),
        'image/object/bbox/xmin': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(
            dtype=tf.float32),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=_DOMAINS_TO_SIZES[domain_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)
