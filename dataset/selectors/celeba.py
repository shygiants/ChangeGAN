""" Selects images that has specific attributes"""

from utils import parse_file, parse_attr, filter_items, append_path

_PREDEFINED_ATTR = {
    'male': ('Male', True),
    'female': ('Male', False),
    'blond_hair': ('Blond_Hair', True),
    'black_hair': ('Black_Hair', True),
}


def check_attr_defined(attributes):
    for attr in attributes:
        assert attr in _PREDEFINED_ATTR


def get_image_files(attrs, directory, attributes_file):
    images = parse_file(attributes_file)
    attrs = parse_attr(attrs, _PREDEFINED_ATTR)
    filtered = filter_items(images, attrs)
    return append_path(directory, filtered)


if __name__ == '__main__':
    images = get_image_files('blond_hair', '', '/Users/SHYBookPro/Desktop/celebA/list_attr_celeba.txt')
    images = get_image_files('black_hair', '', '/Users/SHYBookPro/Desktop/celebA/list_attr_celeba.txt')
    print images
