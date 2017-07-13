
import os
import tensorflow as tf

slim = tf.contrib.slim

_FILE_PATTERN = '%s-*'

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
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
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=_DOMAINS_TO_SIZES[domain_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)
