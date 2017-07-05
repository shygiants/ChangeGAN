""" Selects images that has specific attributes"""
import os
import tensorflow as tf

_PREDEFINED_ATTR = {
    'male': ('Male', True),
    'female': ('Male', False),
    'blond_hair': ('Blond_Hair', True),
    'black_hair': ('Black_Hair', True),
}


def check_attr_defined(attributes):
    for attr in attributes:
        assert attr in _PREDEFINED_ATTR


def get_image_files(attribute, directory, attibutes_file):
    file = tf.gfile.FastGFile(attibutes_file)
    # First line of the file is total number of images
    num_images = int(file.readline())
    print 'There are {} images in dataset'.format(num_images)
    # Second line of the file is field names
    fieldnames = file.readline().strip().split(' ')

    def line_to_dict(line):
        # Parse string
        line = line.strip()
        values = line.split(' ')
        # Remove empty values
        values = filter(lambda val: val != '', values)

        filename = values[0]
        values = values[1:]
        assert len(values) == len(fieldnames)
        dct = dict(zip(fieldnames, values))
        dct['filename'] = filename

        return dct

    lines = file.readlines()
    images = map(line_to_dict, lines)

    # Filter by attribute
    type_attr = type(attribute)
    if type_attr is tuple:
        (key, val) = attribute
    elif type_attr is str:
        (key, val) = _PREDEFINED_ATTR[attribute.lower()]
    else:
        raise TypeError
    filtered = filter(lambda image: bool(int(image[key]) + 1) == val, images)

    print 'There are {} images with {}'.format(len(filtered), attribute)

    # Prepend directory
    return map(lambda image: os.path.join(directory, image['filename']), filtered)

if __name__ == '__main__':
    images = get_image_files('blond_hair', '', '/Users/SHYBookPro/Desktop/celebA/list_attr_celeba.txt')
    print images
