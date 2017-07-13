""" Utils """
import os
import itertools
import tensorflow as tf


def parse_file(filepath,
               val_type=bool,
               fieldnames=None,
               parse_num_items=True,
               parse_fields=True,
               key_item_id='filename',
               validate_fields=True):
    file = tf.gfile.FastGFile(filepath)
    if parse_num_items:
        filename = filepath.split('/')[-1]
        print 'There are {} items in {}'.format(int(file.readline()), filename)
    if parse_fields:
        if fieldnames is None:
            fieldnames = file.readline().strip().split(' ')
            fieldnames = filter(lambda val: val != '', fieldnames)
            if key_item_id is None:
                key_item_id = fieldnames[0]
                fieldnames.remove(key_item_id)
        else:
            # Throw away one line
            file.readline()

    def line_to_dict(line):
        # Parse string
        line = line.strip()
        values = line.split(' ')
        # Remove empty values
        values = filter(lambda val: val != '', values)

        item_id = values[0]
        values = values[1:]
        if val_type == bool:
            values = map(_int2bool, values)
        else:
            values = map(val_type, values)
        if validate_fields:
            assert len(values) == len(fieldnames)
        dct = dict(zip(fieldnames, values))
        dct[key_item_id] = item_id
        return dct

    lines = file.readlines()
    items = map(line_to_dict, lines)

    return items


def parse_attr(attribute, attr_dict):
    type_attr = type(attribute)
    if type_attr is tuple:
        return attribute
    elif type_attr is str:
        if ',' in attribute:
            attrs = attribute.split(',')
            return map(lambda attr: attr_dict[attr.lower()], attrs)
        else:
            return attr_dict[attribute.lower()]
    else:
        raise TypeError


def _int2bool(val):
    val = int(val)
    if val == -1:
        return False
    elif val == 0:
        return None
    elif val == 1:
        return True
    else:
        raise ValueError


def filter_items(items, attrs):
    type_attr = type(attrs)
    if type_attr is tuple:
        (key, val) = attrs
        filtered = filter(lambda image: image[key] == val, items)
    elif type_attr is list:
        def _and(item):
            for (key, val) in attrs:
                if item[key] != val:
                    return False
            return True

        filtered = filter(_and, items)
    else:
        raise TypeError

    print 'There are {} items with {}'.format(len(filtered), attrs)
    return filtered


def append_path(dir, items, key='filename'):
    return map(lambda item: os.path.join(dir, item[key]), items)


def merge(nested_list):
    return list(itertools.chain.from_iterable(nested_list))
