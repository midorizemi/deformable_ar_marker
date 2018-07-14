"""
this program is to calculate affine parameter based-on longitude and latitude
"""

import numpy as np


def oen_parameter():
    return 1.0, 0.0


def asift_basic_parameter_gene():
    """
    ASIFT affine simulation algorithm
    :return:
    """
    yield (1.0, 0.0)
    for t in 2 ** (0.5 * np.arange(1, 6)):
        for phi in np.arange(0, 180, 72.0 / t):
            yield t, phi


def ap_hemisphere_gene(max_t=90, offset_t=10, max_p=180, offset_p=10):
    """
    :param max_t:
    :param offset_t:
    :param max_p:
    :param offset_p:
    :return: tuple of tilt and phi
    """
    yield (1.0, 0.0)
    for t in np.reciprocal(np.cos(np.radians(np.arange(10, max_t, offset_t)))):
        for phi in np.arange(0, max_p, offset_p):
            yield t, phi
