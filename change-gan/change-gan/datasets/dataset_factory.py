"""A factory-pattern class which returns images."""

import celeba

datasets_map = {
    'celeba': celeba,
}


def get_dataset(dataset_name, domain_name, dataset_dir, file_pattern=None, reader=None):
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    return datasets_map[dataset_name].get_dataset(
        domain_name,
        dataset_dir,
        file_pattern,
        reader)
