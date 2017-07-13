""" Selects images """

import os
import ConfigParser

from utils import merge
import tensorflow as tf

config = ConfigParser.ConfigParser()
config.read('config/config.conf')


def get_image_files(attrs):
    images = []
    directory = os.path.join(config.get('crawled', 'image_dir'), attrs)
    for dirname, subdirs, filenames in tf.gfile.Walk(directory):
        if len(subdirs) == 0:
            files = filter(lambda filename: 'jpg' in filename, filenames)
            filepaths = map(lambda file: os.path.join(dirname, file), files)
            images.append(filepaths)
    images = merge(images)
    print 'There are {} items in {}'.format(len(images), directory)
    return images


if __name__ == '__main__':
    get_image_files('clothes')
    get_image_files('models')
