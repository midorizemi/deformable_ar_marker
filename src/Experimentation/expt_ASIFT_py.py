#!/usr/bin/env python

'''
TestScript: {}
ASIFT　のテスト
テンプレート画像同士でチェックする
'''

# Python 2/3 compatibility
from __future__ import print_function

import logging
import os
# built-in modules
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np

from commons import my_file_path_manager as myfsys
from commons.affine_base import affine_detect, match_with_cross
from commons.custom_find_obj import init_feature
from commons.expt_modules import Features
# local modules
from commons.my_common import Timer
from commons.my_file_path_manager import DirNames

logger = logging.getLogger(__name__)

def get_file_path_template_input(testcase, test_sample, template):
    if not template is None:
        templates = myfsys.get_dir_full_path_(DirNames.TEMPLATES.value)
        return os.path.abspath(os.path.join(templates, template))
    if not test_sample is None:
        inputs = myfsys.get_dir_full_path_(DirNames.INPUTS.value)
        return os.path.abspath(os.path.join(inputs, testcase, test_sample))

def setup_expt_directory():
    outputs_dir = myfsys.get_dir_full_path_(DirNames.OUTPUTS.value)
    expt_name, ext = os.path.splitext(os.path.basename(__file__))
    expt_path = os.path.join(outputs_dir, expt_name)
    if os.path.exists(expt_path):
        return expt_path
    os.mkdir(expt_path)
    return expt_path

def read_image(fn):
    import sys
    img = cv2.imread(fn, 0)
    if img is None:
        logger.error('Failed to load fn1:{0}'.format(fn))
        sys.exit(1)
    return img

def asift_detect(detector, fn):
    img = read_image(fn)
    pool = ThreadPool(processes=cv2.getNumberOfCPUs())
    with Timer('Detection with [ ASIFT ]'):
        splt_kp, splt_desc = affine_detect(detector, img, pool=pool, simu_param='asift')
    return img, splt_kp, splt_desc

def asfit():
    pass

if __name__ == '__main__':
    expt_path = setup_expt_directory()
    logging.basicConfig(filename=os.path.join(expt_path, 'log.txt'), level=logging.DEBUG)
    logger.info(__doc__.format(os.path.basename(__file__)))
    a = os.listdir(myfsys.get_dir_full_path_(DirNames.TEMPLATES.value))
    a.pop(a.index('mesh_label.png'))
    detector, matcher = init_feature(Features.SIFT.name)
    for testcase in a:
        logger.info(testcase + 'の場合')
        fn = myfsys.get_template_file_full_path_(testcase)
        imgQ, kpQ, descQ = asift_detect(detector, fn)
        imgT, kpT, descT = asift_detect(detector, fn)
        logger.info('imgQ - %d features, imgT - %d features' % (len(kpQ), len(kpT)))
        with Timer('matching'):
            pQ, pT, pairs = match_with_cross(matcher, descQ, kpQ, descT, kpT)
        if len(pQ) >= 4:
            H, status = cv2.findHomography(pQ, pT, cv2.RANSAC, 5.0)
            logger.info('%d / %d  inliers/matched' % (np.sum(status), len(status)))
            # do not draw outliers (there will be a lot of them)
            pairs = [kpp for kpp, flag in zip(pairs, status) if flag]
        else:
            H, status = None, None
            logger.info('%d matches found, not enough for homography estimation' % len(pairs))
