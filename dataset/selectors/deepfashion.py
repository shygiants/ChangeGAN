""" Selects images that has specific attributes"""

from utils import parse_file, parse_attr, filter_items, append_path

_PREDEFINED_ATTR = {
    'top': ('clothes_type', 1),
    'flat': ('pose_type', 6),
    'front': ('pose_type', 1),
    'zoom': ('variation_type', 4)
}


def get_image_files(attrs, directory, attributes_file, validate_fields=False):
    images = parse_file(attributes_file, val_type=int, key_item_id=None, validate_fields=validate_fields)
    attrs = parse_attr(attrs, _PREDEFINED_ATTR)
    filtered = filter_items(images, attrs)
    return append_path(directory, filtered, key='image_name')


if __name__ == '__main__':
    images = get_image_files('top,flat',
                             '/Users/SHYBookPro/Desktop/deepFashion/inshop',
                             '/Users/SHYBookPro/Desktop/deepFashion/inshop/Anno/list_bbox_inshop.txt')
    images = get_image_files('top,front',
                             '/Users/SHYBookPro/Desktop/deepFashion/inshop',
                             '/Users/SHYBookPro/Desktop/deepFashion/inshop/Anno/list_bbox_inshop.txt')
    # images = get_image_files('top,zoom',
    #                          '/Users/SHYBookPro/Desktop/deepFashion/landmark',
    #                          '/Users/SHYBookPro/Desktop/deepFashion/landmark/Anno/list_landmarks.txt',
    #                          validate_fields=False)
    print images[0:3]
