""" Utils """

import os
import ConfigParser
import urllib
import itertools

config = ConfigParser.ConfigParser()
config.read('config/config.conf')


def img2src(imgs):
    return map(lambda img: img.get_attribute('src'), imgs)


def img2srcset(imgs):
    return map(lambda img: img.get_attribute('srcset'), imgs)


def download_images(imgs, dir, start=0):
    dir = os.path.join(config.get('dataset', 'image_target_dir'), dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    for i, img in enumerate(imgs):
        urllib.urlretrieve(str(img), os.path.join(dir, '{}.jpg'.format(i + start)))


def merge(nested_list):
    return list(itertools.chain.from_iterable(nested_list))
