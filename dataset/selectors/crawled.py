""" Selects images """

import os
from utils import merge
import tensorflow as tf


def get_image_files(attrs, directory):
    images = []
    directory = os.path.join(directory, attrs)
    for dirname, subdirs, filenames in tf.gfile.Walk(directory):
        if len(subdirs) == 0:
            files = filter(lambda filename: 'jpg' in filename, filenames)
            filepaths = map(lambda file: os.path.join(dirname, file), files)
            images.append(filepaths)
    images = merge(images)
    print 'There are {} items in {}'.format(len(images), directory)
    return images


if __name__ == '__main__':
    get_image_files('clothes', '/Users/SHYBookPro/Desktop/clothes')
    get_image_files('models', '/Users/SHYBookPro/Desktop/clothes')
