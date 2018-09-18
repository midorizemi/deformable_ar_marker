#!/usr/bin/env python

'''
SPLIT_ASIFRT　のテスト{}
メッシュ領域毎のメッシュ検出性能を測定
'''

# Python 2/3 compatibility
from __future__ import print_function

import logging.handlers
import os
# built-in modules
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np

from commons import expt_modules as emod, my_file_path_manager as myfsys
from commons.custom_find_obj import calclate_Homography
from commons.custom_find_obj import init_feature
from commons.my_common import Timer
from commons.my_common import format4pickle_pairs
import commons.template_info as TmpInf
from make_database import split_affinesim_combinable as slac

# local modules

logger = logging.getLogger(__name__)

def make_logger():
    # logging.basicConfig(filename=os.path.join(expt_path, 'log.txt'), level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG,
                        datefmt='%m-%d %H:%M')
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(lineno)d:%(levelname)5s\n  |>%(message)s')
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(level=logging.DEBUG)
    consoleHandler.setFormatter(formatter)
    timeRotationHandler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(expt_path, prefix + fn1 + '_log.txt'),
        when='h',
        interval=24,
        backupCount=3,
        encoding='utf8'
    )
    timeRotationHandler.setLevel(level=logging.INFO)
    timeRotationHandler.setFormatter(formatter)
    logging.getLogger(__name__).addHandler(timeRotationHandler)
    logging.getLogger(__name__).addHandler(consoleHandler)
    logging.getLogger(__name__).setLevel(level=logging.DEBUG)
    # logging.getLogger('commons').setLevel(level=logging.DEBUG)
    # logging.getLogger('commons').addHandler(timeRotationHandler)
    # logging.getLogger('commons').addHandler(consoleHandler)
    logging.getLogger('commons.affine_base').setLevel(level=logging.WARNING)
    logging.getLogger('commons.affine_base').addHandler(timeRotationHandler)
    logging.getLogger('commons.affine_base').addHandler(consoleHandler)
    logging.getLogger('commons.custom_find_obj').setLevel(level=logging.WARNING)
    logging.getLogger('commons.custom_find_obj').addHandler(timeRotationHandler)
    logging.getLogger('commons.custom_find_obj').addHandler(consoleHandler)
    logging.getLogger('commons.my_common').setLevel(level=logging.WARNING)
    logging.getLogger('commons.my_common').addHandler(timeRotationHandler)
    logging.getLogger('commons.my_common').addHandler(consoleHandler)
    # logging.getLogger('make_database').setLevel(level=logging.DEBUG)
    # logging.getLogger('make_database').addHandler(timeRotationHandler)
    # logging.getLogger('make_database').addHandler(consoleHandler)
    logging.getLogger('make_database.split_affinesim').setLevel(level=logging.DEBUG)
    logging.getLogger('make_database.split_affinesim').addHandler(timeRotationHandler)
    logging.getLogger('make_database.split_affinesim').addHandler(consoleHandler)
    logging.getLogger('expt_modules').setLevel(level=logging.DEBUG)
    logging.getLogger('expt_modules').addHandler(timeRotationHandler)
    logging.getLogger('expt_modules').addHandler(consoleHandler)
    logging.getLogger('my_file_path_manager').setLevel(level=logging.DEBUG)
    logging.getLogger('my_file_path_manager').addHandler(timeRotationHandler)
    logging.getLogger('my_file_path_manager').addHandler(consoleHandler)


def check_not_None(arg):
    if arg is not None:
        return len(arg)
    else:
        return 0

def draw_meshes(fn2, testset_full_path):
    fn, ext = os.path.splitext(fn2)
    testcase_full_path = os.path.join(testset_full_path, fn2)
    imgT = cv2.imread(testcase_full_path, 0)
    if imgT is None:
        logger.info('Failed to load fn2:', testcase_full_path)
        raise ValueError('Not found the file')
    logger.info("Using Training: {}".format(fn2))

    #loading
    import joblib
    dump_match_testcase_dir = myfsys.setup_output_directory(dump_match_dir, fn)
    Hs = joblib.load(os.path.join(dump_match_testcase_dir, 'Hs.pikle'))
    statuses = joblib.load(os.path.join(dump_match_testcase_dir, 'statuses.pikle'))
    import pickle
    with open(os.path.join(dump_match_testcase_dir, 'pairs.pickle'), 'rb') as f:
        index_pairs = pickle.load(f)
    pairs = list(list(cv2.KeyPoint(x=p[0][0], y=p[0][1], _size=p[1], _angle=p[2],
                            _response=p[3], _octave=p[4], _class_id=p[5]) for p in pair) for pair in index_pairs)

    vis = slac.draw_matches_for_meshes(imgQ, imgT, temp_inf=temp_inf, Hs=Hs,
                                       list_merged_mesh_id=list_merged_mesh_id, merged_map=merged_map)

    cv2.imwrite(os.path.join(detected_dir, fn2 + '.png'), vis)

    viw = slac.explore_match_for_meshes('affine find_obj', imgQ, imgT, pairs,
                                        temp_inf=temp_inf, Hs=Hs, status=statuses,
                                        list_merged_mesh_id=list_merged_mesh_id, merged_map=merged_map)

    cv2.imwrite(os.path.join(line_dir, fn2 + '.png'), viw)
    cv2.destroyAllWindows()

def dump_matching_result(fn2, testset_full_path):
    #外部ファイルに出力する
    #個々のファイル毎にデータを出力する
    fn, ext = os.path.splitext(fn2)
    testcase_full_path = os.path.join(testset_full_path, fn2)
    imgT = cv2.imread(testcase_full_path, 0)
    if imgT is None:
        logger.info('Failed to load fn2:', testcase_full_path)
        raise ValueError('Not found the file')
    logger.info("Using Training: {}".format(fn2))

    pool = ThreadPool(processes=cv2.getNumberOfCPUs())
    with Timer('Detection'):
        kpT, descT = slac.affine_detect(detector, imgT, pool=pool, simu_param='test')
    logger.info('imgQ - %d features, imgT - %d features' % (slac.count_keypoints(splt_kpQ), len(kpT)))

    with Timer('matching'):
        mesh_pQ, mesh_pT, mesh_pairs = slac.match_with_cross(matcher, m_sdQ, m_skQ, descT, kpT)

    index_mesh_pairs = format4pickle_pairs(mesh_pairs)
    import joblib
    dump_match_testcase_dir = myfsys.setup_output_directory(dump_match_dir, fn)
    joblib.dump(mesh_pQ, os.path.join(dump_match_testcase_dir, 'mesH_pQ.pikle'), compress=True)
    joblib.dump(mesh_pT, os.path.join(dump_match_testcase_dir, 'mesH_pT.pikle'), compress=True)
    import pickle
    with open(os.path.join(dump_match_testcase_dir, 'mesh_pairs.pickle'), 'wb') as f:
        pickle.dump(index_mesh_pairs, f)
        f.close()
    # for i, mesh_pair in enumerate(index_mesh_pairs):
    #     joblib.dump(mesh_pair, os.path.join(dump_detected_testcase_dir, "mesh_pairs_{0:02d}.pikle".format(i)),
    #                 compress=True)


    with Timer('estimation'):
        Hs, statuses, pairs = slac.calclate_Homography4splitmesh(mesh_pQ, mesh_pT, mesh_pairs, median=median)
    joblib.dump(Hs, os.path.join(dump_match_testcase_dir, 'Hs.pikle'), compress=True)
    joblib.dump(statuses, os.path.join(dump_match_testcase_dir, 'statuses.pikle'), compress=True)
    index_pairs = tuple(
        tuple((p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in pair) for pair in pairs)
    with open(os.path.join(dump_match_testcase_dir, 'pairs.pickle'), 'wb') as f:
        pickle.dump(index_pairs, f)



if __name__ == '__main__':
    import sys, getopt, os
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    task, fn1, prefix = args
    expt_path = myfsys.setup_expt_directory(os.path.basename(__file__))
    make_logger()

    logger.info(__doc__.format(os.path.basename(__file__)))
    template_full_fn = myfsys.get_template_file_full_path_(fn1)
    imgQ = cv2.imread(template_full_fn, 0)
    if imgQ is None:
        print('Failed to load fn1:', template_full_fn)
        sys.exit(1)
    print("Using Query: {}".format(fn1))

    detector, matcher = init_feature(emod.Features.SIFT.name)
    if detector is None:
        print('unknown feature:', emod.Features.SIFT.name)
        sys.exit(1)
    print('Using :', emod.Features.SIFT.name)

    scols = 8
    srows = 8
    h, w = imgQ.shape[:2]
    template_fn, ext = os.path.splitext(fn1)
    template_information = {"_fn": "tmp.png", "template_img": template_fn,
                            "_cols": w, "_rows": h, "_scols": scols, "_srows": srows, "_nneighbor": 4}
    temp_inf = TmpInf.TemplateInfo(**template_information)

    try:
        with Timer('Lording pickle'):
            splt_kpQ, splt_descQ = slac.affine_load_into_mesh(template_fn, temp_inf.get_splitnum())
    except ValueError as e:
        print(e.args)
        print('If you need to save {} to file as datavase. ¥n'
              + ' Execute makedb/make_split_combine_featureDB_from_templates.py')
        with Timer('Detection and dividing'):
            splt_kpQ, splt_descQ = slac.affine_detect_into_mesh(detector, temp_inf.get_splitnum(),
                                                           imgQ, simu_param='default')

    m_skQ, m_sdQ, m_k_num, merged_map = slac.combine_mesh_compact(splt_kpQ, splt_descQ, temp_inf)
    list_merged_mesh_id = list(set(np.ravel(merged_map)))
    median = np.nanmedian(m_k_num)

    expt_name = os.path.basename(expt_path)

    keyargs = {'prefix_shape': prefix, 'template_fn': template_fn}
    testset_full_path = myfsys.get_dir_full_path_testset('cgs', **keyargs)
    testset_name = os.path.basename(testset_full_path)
    logger.debug('testset_name is {}'.format(testset_name))
    logger.info('Test Set:{}'.format(testset_name))
    testcase_fns = os.listdir(testset_full_path)
    testcase_fns.sort()
    # logger.debug(testcase_fns)


    output_dir = myfsys.setup_output_directory(expt_name, testset_name)
    detected_dir = myfsys.setup_output_directory(output_dir, 'detected_mesh')
    line_dir = myfsys.setup_output_directory(output_dir, 'dmesh_line')
    dump_match_dir = myfsys.setup_output_directory(output_dir, 'dump_match_dir')
    dump_detected_dir = myfsys.setup_output_directory(output_dir, 'dump_detected_dir')



    test = 0
    for fn2 in testcase_fns:

        try:
            if task == 'draw':
                draw_meshes(fn2, testset_full_path)
            if task == 'dump':
                dump_matching_result(fn2, testset_full_path)
        except ValueError as e:
            import traceback
            traceback.print_exc()
            logger.warning(e.args)
            logger.info('====CONTINUE====')
            continue
        finally:
            pass
            if test<5:
                test += 1
            else:
                sys.exit(0)


        # fn, ext = os.path.splitext(fn2)
        # testcase_full_path = os.path.join(testset_full_path, fn2)
        # imgT = cv2.imread(testcase_full_path, 0)
        # if imgT is None:
        #     logger.info('Failed to load fn2:', testcase_full_path)
        #     logger.info('====CONTINUE====')
        #     continue
        # logger.info("Using Training: {}".format(fn2))
        #
        # pool = ThreadPool(processes=cv2.getNumberOfCPUs())
        # with Timer('Detection'):
        #     kpT, descT = slac.affine_detect(detector, imgT, pool=pool, simu_param='test')
        # logger.info('imgQ - %d features, imgT - %d features' % (slac.count_keypoints(splt_kpQ), len(kpT)))
        #
        # with Timer('matching'):
        #     mesh_pQ, mesh_pT, mesh_pairs = slac.match_with_cross(matcher, m_sdQ, m_skQ, descT, kpT)
        #
        # # Hs, statuses, pairs = calclate_Homography4splitmesh(mesh_pQ, mesh_pT, mesh_pairs)
        # # with Timer('estimation'):
        # #     Hs, statuses, pairs = slac.calclate_Homography4splitmesh(mesh_pQ, mesh_pT, mesh_pairs)
        # Hs = []
        # statuses = []
        # pairs = []
        # def f(*pQpTp):
        #     inlier_pairs, H, status = calclate_Homography(pQpTp[0], pQpTp[1], pQpTp[2])
        #     Hs.append(H)
        #     pairs.extend(inlier_pairs)
        #     if status is None:
        #         status = []
        #     statuses.extend(status)
        #     return [len(inlier_pairs), len(status), len(pQpTp[2])]
        #
        # with Timer('estimation'):
        #     vals_list = np.array(list(map(f, mesh_pQ, mesh_pT, mesh_pairs)))
        #
        # vis = slac.draw_matches_for_meshes(imgQ, imgT, temp_inf=temp_inf, Hs=Hs,
        #                                    list_merged_mesh_id=list_merged_mesh_id, merged_map=merged_map)
        #
        # cv2.imwrite(os.path.join(detected_dir, fn2 + '.png'), vis)
        #
        # viw = slac.explore_match_for_meshes('affine find_obj', imgQ, imgT, pairs,
        #                                temp_inf=temp_inf, Hs=Hs, status=statuses,
        #                                list_merged_mesh_id=list_merged_mesh_id, merged_map=merged_map)
        #
        # cv2.imwrite(os.path.join(line_dir, fn2 + '.png'), viw)
        # cv2.destroyAllWindows()


    # keywords = list(map(lambda z: os.path.splitext(z)[0], testcase_fns))
    # dictionary = dict(zip(keywords, vals_list))
    # np.savez_compressed(os.path.join(output_dir, testset_name), **dictionary)

