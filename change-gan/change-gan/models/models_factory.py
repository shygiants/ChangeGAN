"""A factory-pattern class which returns models."""

import autoconverter
import change_gan

models_map = {
    'autoconverter': autoconverter,
    'change_gan': change_gan,
}


def get_model(model_name):
    if model_name not in models_map:
        raise ValueError('Name of dataset unknown %s' % model_name)
    return models_map[model_name]
