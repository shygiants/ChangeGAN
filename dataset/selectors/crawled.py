""" Selects images """

import os
import tensorflow as tf
from utils import parse_file, parse_attr, filter_items, append_path


def get_image_files(attrs, directory):
    num_images = 0
    for dirname, subdirs, filenames in tf.gfile.Walk(os.path.join(directory, attrs)):
        if len(subdirs) == 0:
            num_images += len(filter(lambda filename: 'jpg' in filename, filenames))
    print num_images


if __name__ == '__main__':
    get_image_files('clothes', '/Users/SHYBookPro/Desktop/clothes')
    get_image_files('models', '/Users/SHYBookPro/Desktop/clothes')
