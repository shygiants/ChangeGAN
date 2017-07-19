""" Selects images that has specific attributes"""

import os
import ConfigParser
from utils import parse_file, parse_attr, filter_items, append_path, bbox

config = ConfigParser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config/config.conf'))

_PREDEFINED_ATTR = {
    'top': ('clothes_type', 1),
    'bottom': ('clothes_type', 2),
    'flat': ('pose_type', 6),
    'front': ('pose_type', 1),
    'zoom': ('variation_type', 4)
}


def get_image_files(attrs, validate_fields=False):
    images = parse_file(config.get('deepfashion', 'attributes_file'),
                        val_type=int, key_item_id=None, validate_fields=validate_fields)
    attrs = parse_attr(attrs, _PREDEFINED_ATTR)
    filtered = filter_items(images, attrs)
    return append_path(config.get('deepfashion', 'image_dir'), filtered, key='image_name')


def get_image_with_bbox(attrs):
    images = parse_file(config.get('deepfashion', 'attributes_file'),
                        val_type=int, key_item_id=None, validate_fields=False)
    attrs = parse_attr(attrs, _PREDEFINED_ATTR)
    filtered = filter_items(images, attrs)

    image_files = append_path(config.get('deepfashion', 'image_dir'), filtered, key='image_name')
    boxes = bbox(filtered)

    return image_files, boxes


if __name__ == '__main__':
    get_image_with_bbox('front')

    # images = get_image_files('top,flat')
    # images = get_image_files('top,front')
    # print images[0:3]
