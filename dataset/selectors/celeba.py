""" Selects images that has specific attributes"""

import ConfigParser
from utils import parse_file, parse_attr, filter_items, append_path

config = ConfigParser.ConfigParser()
config.read('config/config.conf')

_PREDEFINED_ATTR = {
    'male': ('Male', True),
    'female': ('Male', False),
    'blond_hair': ('Blond_Hair', True),
    'black_hair': ('Black_Hair', True),
}


def get_image_files(attrs):
    images = parse_file(config.get('celeba', 'attributes_file'))
    attrs = parse_attr(attrs, _PREDEFINED_ATTR)
    filtered = filter_items(images, attrs)
    return append_path(config.get('celeba', 'image_dir'), filtered)


if __name__ == '__main__':
    images = get_image_files('blond_hair')
    images = get_image_files('black_hair')
    print images
