"""
Experimentation for split-ASIFT which is detection and matching
"""
from __future__ import print_function

import logging
import os
# built-in modules
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np

from commons import expt_modules as emod, my_file_path_manager as myfsys
from commons.affine_base import affine_detect
from commons.custom_find_obj import calclate_Homography, draw_matches_for_meshes
from commons.custom_find_obj import explore_match_for_meshes
from commons.find_obj import init_feature
# local modules
from commons.my_common import Timer
from make_database import split_affinesim as spltA

logger = logging.getLogger(__name__)

def read_images(fnQ, fnT, logger):
    import sys
    imgQ = cv2.imread(fnQ, 0)
    imgT = cv2.imread(fnT, 0)
    if imgQ is None:
        logger.error('Failed to load fn1:{0}'.format(fnQ))
        sys.exit(1)

    if imgT is None:
        logger.error('Failed to load fn2:{0}'.format(fnT))
        sys.exit(1)
    return imgQ, imgT

def read_image(fn):
    import sys
    img = cv2.imread(fn, 0)
    if img is None:
        logger.error('Failed to load fn1:{0}'.format(fn))
        sys.exit(1)
    return img

def setup(expt_names):
    """実験開始用ファイル読み込み"""
    if not os.path.exists(os.path.join(myfsys.getd_outpts(expt_names), 'log')):
        os.makedirs(os.path.join(myfsys.getd_outpts(expt_names), 'log'))
    # create file handler which logs even debug messages
    fh = logging.FileHandler(myfsys.getf_log(expt_names, expt_names[2] + '.log'))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def detect(detector, fn, splt_num=64, simu_type="default"):
    full_fn = myfsys.get_template_file_full_path_(fn)
    img = read_image(full_fn)
    cv2.imshow('hoge', img)
    with Timer('Detection with [ ' + simu_type + ' ]'):
        splt_kp, splt_desc = spltA.affine_detect_into_mesh(detector, splt_num, img, simu_param=simu_type)
    return img, splt_kp, splt_desc

def detect_sift(detector, set_fn, logger, pool=None):
    fnQ, testcase, fnT = set_fn
    full_fnT = myfsys.getf_input(testcase, fnT)
    imgT = read_image(full_fnT)
    with Timer('Detection with SFIT'):
        kpT, descT = affine_detect(detector, imgT, pool=pool, simu_param='sift')
    return imgT, kpT, descT

def get_templatef_inputf_outputd(expt_name, testcase, test_sample, template):
    if not template is None:
        templates = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'templates'))
        return os.path.abspath(os.path.join(templates, template))
    if not test_sample is None:
        inputs = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data", "inputs"))
        return os.path.abspath(os.path.join(inputs, testcase, test_sample))
    else:
        outputs = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data", "outputs"))
        return os.path.abspath(os.path.join(outputs, expt_name, testcase))

def match(matcher, kpQ, descQ, kpT, descT, id):
    with Timer('matching', logger):
        mesh_pQ, mesh_pT, mesh_pairs = spltA.match_with_cross(matcher, descQ, kpQ, descT, kpT)
    return mesh_pQ, mesh_pT, mesh_pairs, id

def estimate_mesh(mesh_pQ, mesh_pT, mesh_pairs):
    inliner_matches = []
    for pQ, pT, pairs in zip(mesh_pQ, mesh_pT, mesh_pairs):
        pairs, H, status = calclate_Homography(pQ, pT, pairs)

def detect_and_match(detector, matcher, set_fn, splt_num=64, simu_type="default"):
    """
    SplitA実験
    set_fn:
    """
    fnQ, testcase, fnT = set_fn
    def get_expt_names():
        tmpf, tmpext = os.path.splitext(fnT)
        return (os.path.basename(__file__), testcase, tmpf)
    expt_names = get_expt_names()
    logger = setup(expt_names)
    logger.info(__doc__)

    full_fnQ = myfsys.getf_template((fnQ,))
    full_fnT = myfsys.getf_input(testcase, fnT)
    imgQ, imgT = read_images(full_fnQ, full_fnT, logger)

    pool = ThreadPool(processes=cv2.getNumberOfCPUs())
    with Timer('Detection with SPLIT-ASIFT', logger):
        splt_kpQ, splt_descQ = spltA.affine_detect_into_mesh(detector, splt_num, imgQ, simu_param=simu_type)
    with Timer('Detection with SFIT', logger):
        kpT, descT = affine_detect(detector, imgT, pool=pool, simu_param='test')
    logger.info('imgQ - {0} features, imgT - {1} features'.format(spltA.count_keypoints(splt_kpQ), len(kpT)))

    with Timer('matching', logger):
        mesh_pQ, mesh_pT, mesh_pairs = spltA.match_with_cross(matcher, splt_descQ, splt_kpQ, descT, kpT)

    Hs = []
    statuses = []
    kp_pairs_long = []
    Hs_stable = []
    kp_pairs_long_stable = []
    for pQ, pT, pairs in zip(mesh_pQ, mesh_pT, mesh_pairs):
        pairs, H, status = calclate_Homography(pQ, pT, pairs)
        Hs.append(H)
        statuses.append(status)
        if status is not None and not len(status) == 0 and np.sum(status)/len(status) >= 0.4:
            Hs_stable.append(H)
        else:
            Hs_stable.append(None)
        for p in pairs:
            kp_pairs_long.append(p)
            if status is not None and not len(status) == 0 and np.sum(status)/len(status) >= 0.4:
                kp_pairs_long_stable.append(p)

    vis = draw_matches_for_meshes(imgQ, imgT, Hs=Hs)
    cv2.imwrite(myfsys.getf_output(expt_names, 'meshes.png'), vis)

    visS = draw_matches_for_meshes(imgQ, imgT, Hs=Hs_stable)
    cv2.imwrite(myfsys.getf_output(expt_names, 'meshes_stable.png'), visS)

    viw = explore_match_for_meshes('affine find_obj', imgQ, imgT, kp_pairs_long_stable, Hs=Hs_stable)
    cv2.imwrite(myfsys.getf_output(expt_names, 'meshes_and_keypoints_stable.png'), viw)

    return vis, visS, viw


if __name__ == '__main__':
    expt_path = myfsys.setup_expt_directory(os.path.basename(__file__))
    logging.basicConfig(filename=os.path.join(expt_path, 'log.txt'), level=logging.DEBUG)
    a = os.listdir(myfsys.get_dir_full_path_(myfsys.DirNames.TEMPLATES.value))
    a.pop(a.index('mesh_label.png'))
    detector, matcher = init_feature(emod.Features.SIFT.name)
    for template in a:
        template_name, ext = template.split('.')
        print(template_name)
        fn = myfsys.get_template_file_full_path_(template)
        imgQ, kpQ, descQ = detect(detector, fn, splt_num=64, simu_type='asift')
        #実験は平面のみ
        subj_dir = emod.PrefixShapes.PL.value + template_name
        testsets = os.listdir(myfsys.get_inputs_dir_full_path(subj_dir))
        for testcase in testsets:
            print(myfsys.getf_input(subj_dir, testcase))


        # fnQ = ('qrmarker.png', 'menko.png', 'nabe.png', 'nabe.png', 'nabe.png')
        # dirT = ('unittest', 'crv_menko', 'pics_nabe', 'real-crv_nabe', 'mltf_nabe')
        # fnT = ('smpl_1.414214_152.735065.png', 'smpl_2.828427_152.735065.png', 'IMG_1118.JPG', 'my_photo-5.jpg', 'smpl_1.414214_152.735065.png')

    # set_fn = (fnQ[0], dirT[0], fnT[0])
    # vis, visS, viw = detect_and_match(detector, matcher, set_fn)
    # set_fn = (fnQ[1], dirT[1], fnT[1])
    # vis, visS, viw = detect_and_match(detector, matcher, set_fn)
    # set_fn = (fnQ[2], dirT[2], fnT[2])
    # vis, visS, viw = detect_and_match(detector, matcher, set_fn)
    # set_fn = (fnQ[3], dirT[3], fnT[3])
    # vis, visS, viw = detect_and_match(detector, matcher, set_fn)
    #set_fn = (fnQ[4], dirT[4], fnT[4])
    #vis, visS, viw = detect_and_match(detector, matcher, set_fn)
    # cv2.imshow('view weak meshes', vis)
    # cv2.imshow('view stable meshes', visS)
    # cv2.imwrite('qr1_mesh_line.png', viw)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


