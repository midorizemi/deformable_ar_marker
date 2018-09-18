#!/usr/bin/env python

'''
SPLIT_ASIFRT　のテスト{}
メッシュ領域毎のメッシュ検出性能を測定
'''

# Python 2/3 compatibility
from __future__ import print_function

import logging.handlers
# built-in modules
from multiprocessing.pool import ThreadPool

#プロット系
import os
import sys
if os.getenv('DISPLAY') is None:
    #もし，SSHサーバーサイドで実行するなら，
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    plt.switch_backend('pdf')
    from matplotlib.backends.backend_pdf import PdfPages
else:
    import matplotlib
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import pickle
import numpy as np

import pandas

from commons import expt_modules as emod, my_file_path_manager as myfsys
from commons.custom_find_obj import calclate_Homography
from commons.custom_find_obj import init_feature
from commons.my_common import Timer
from make_database import split_affinesim_combinable as slac
from make_database import asift

# local modules

logger = logging.getLogger(__name__)

def get_img(fn):
    img = cv2.imread(fn, 0)
    if img is None:
        raise ValueError('Failed to load fn1:', fn)
    return img

def get_detector_matchier(feature='sift'):
    detector, matcher = init_feature(feature)
    if detector is None:
        print('unknown feature:', feature)
    return detector, matcher

if __name__ == "__main__":
    template_dir_path = myfsys.get_dir_full_path_(myfsys.DirNames.TEMPLATES.value)
    dump_dir = 'dump_features'
    a = myfsys.make_list_template_filename()
    feature = 'sift'
    def mkdir_dump_dir(*args, **kwargs):
        path = os.path.join(kwargs.get('base_dir', ''), *args)
        if os.path.exists(path):
            print(path + 'is exist')
            return path, True
        os.makedirs(path, exist_ok=True)
        print('make dir_path_full')
        return path, False

    for template_fn in a:
        fn, ext = os.path.splitext(template_fn)
        path_dump_dir, isExist = mkdir_dump_dir(template_dir_path, dump_dir, fn)
        if isExist:
            print('すでにキーポイントを生成済み')
            continue
        try:
            detector, matcher = get_detector_matchier(feature)
            print('using: ', feature)
        except ValueError as e:
            print(e)
            sys.exit(1)

        template_full_fn = myfsys.get_template_file_full_path_(template_fn)
        try:
            imgQ = get_img(template_full_fn)
        except ValueError as e:
            print(e)
            print('テンプレートがないだけなので続ける')
            continue
        template_information = {"_fn":"tmp.png", "template_img":template_fn,
                                "_cols":800, "_rows":600, "_scols":8, "_srows":8, "_nneighbor":4}
        temp_inf = slac.TmpInf(**template_information)
        pool = ThreadPool(processes=cv2.getNumberOfCPUs())
        with Timer('calculate Keypoints Descriptors'):
            kp, des = asift.affine_detect(detector, imgQ, pool=pool)
        ##キーポイントを出力する
        index = []
        for p in kp:
            temp = (p.pt, p.size, p.angle, p.response, p.octave, p.class_id)
            index.append(temp)
        pickle_path = myfsys.get_pikle_path(fn)
        with open(pickle_path, mode='wb') as f:
            pickle.dump((index, des), f)
