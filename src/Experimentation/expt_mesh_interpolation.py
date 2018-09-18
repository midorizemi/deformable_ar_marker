from make_database import split_affinesim as splta
from commons.my_common import load_pickle_mesh_matchepairs
from commons.my_common import load_pickle_matchepairs
import os
from make_database import mesh_interpolation as m_in

def loading_img(fn1_full, fn2_full, feature_name):
    imgQ = splta.cv2.imread(fn1_full, 0)
    detector, matcher = splta.init_feature(feature_name)

    if imgQ is None:
        print('Failed to load fn1:', fn1_full)
        sys.exit(1)
    print("MARKER: {}".format(fn1_full))

    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)
    print("FEATURE: {}".format(feature_name))
    imgT = splta.cv2.imread(fn2_full, 0)
    if imgT is None:
        print('Failed to load fn2:', fn2_full)
        sys.exit(1)
    print("INPUT: {}".format(fn2_full))

    return imgQ, imgT, detector, matcher

def get_split_keypoint_detector(_template_fn, _temp_inf, _detector, _imgQ):
    try:
        with splta.Timer('Lording pickle'):
            splt_kpQ, splt_descQ = splta.affine_load_into_mesh(_template_fn, _temp_inf.get_splitnum())
    except ValueError as e:
        print(e.args)
        print('If you need to save {} to file as datavase. ¥n'
              + ' Execute makedb/make_split_combine_featureDB_from_templates.py')
        with splta.Timer('Detection and dividing'):
            splt_kpQ, splt_descQ = splta.affine_detect_into_mesh(_detector, _temp_inf.get_splitnum(),
                                                                 _imgQ, simu_param='asift')

    return splt_kpQ, splt_descQ


def get_matched_points(dumped_exdir, testset_name, fn, sdscQ, skpQ, dscT, kpT):
    try:
        with splta.Timer('Loarding matching pickle'):
            mesh_pQ, mesh_pT, mesh_pairs = m_in.load_pickle_match_with_cross(dumped_exdir, testset_name, fn)
            # mesh_pQ, mesh_pT, mesh_pairs = splta_c.match_with_cross(matcher, splt_descQ, splt_kpQ, descT, kpT)
    except:
        print('Failed Load matching result')
        with splta.Timer('matching'):
            mesh_pQ, mesh_pT, mesh_pairs = splta.match_with_cross(matcher, sdscQ, skpQ, dscT, kpT)
    return mesh_pQ, mesh_pT, mesh_pairs


def get_homographies(dumped_exdir, testset_name, fn, mesh_pQ, mesh_pT, mesh_pairs):
    try:
        with splta.Timer('Loading estimation result'):
            Hs, statuses, pairs = m_in.load_pickle_calclate_Homography4splitmesh(dumped_exdir, testset_name, fn)
            # Hs, statuses, pairs = splta_c.calclate_Homography4splitmesh(mesh_pQ, mesh_pT, mesh_pairs, median=median)
    except:
        print('Failed loading estimated mesh')
        with splta.Timer('estimation'):
            Hs, statuses, pairs = splta.calclate_Homography4splitmesh(mesh_pQ, mesh_pT, mesh_pairs, median=median)

    return Hs, statuses, pairs
if __name__ == '__main__':
    print(__doc__)

    import sys
    import getopt

    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'sift')
    try:
        fn1_full, fn2_full, pr, testset_dir_full = args
    except:
        dir_path_full = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        fn1_full = os.path.abspath(os.path.join(dir_path_full, 'data/templates/qrmarker.png'))
        fn2_full = os.path.abspath(os.path.join(dir_path_full, 'data/inputs/cgs/mltf_qrmarker/057_070-200.png'))
        testset_dir_full = os.path.abspath(os.path.join(dir_path_full, 'data/inputs/cgs/mltf_qrmarker'))
        pr = "mltf_"

    imgQ, imgT, detector, matcher = loading_img(fn1_full, fn2_full, feature_name)
    scols = 8
    srows = 8
    h, w = imgQ.shape[:2]
    template_fn, ext = os.path.splitext(os.path.basename(fn1_full))
    template_information = {"_fn": "tmp.png", "template_img": template_fn,
                            "_cols": w, "_rows": h, "_scols": scols, "_srows": srows, "_nneighbor": 4}
    temp_inf = splta.TmpInf(**template_information)

    skpQ, sdscQ = get_split_keypoint_detector(template_fn, temp_inf, detector, imgQ)
    mesh_k_num = splta.np.array([len(keypoints) for keypoints in skpQ]).reshape(temp_inf.get_mesh_shape())
    median = splta.np.nanmedian(mesh_k_num)

    pool = splta.ThreadPool(processes=splta.cv2.getNumberOfCPUs())
    with splta.Timer('Detection'):
        kpT, dscT = splta.affine_detect(detector, imgT, pool=pool, simu_param='test')
    print('imgQ - %d features, imgT - %d features' % (splta.count_keypoints(skpQ), len(kpT)))

    fn, ext = os.path.splitext(os.path.basename(fn2_full))
    testset_name = os.path.basename(os.path.dirname(fn2_full))
    dumped_exdir = "expt_split_affinesim"
    # dumped_exdir = "expt_split_affinesim_conbine"
    mesh_pQ, mesh_pT, mesh_pairs = get_matched_points(dumped_exdir, testset_name, fn, sdscQ, skpQ, dscT, kpT)

    Hs, statuses, pairs = get_homographies(dumped_exdir, testset_name, fn, mesh_pQ, mesh_pT, mesh_pairs)


    # 実験用出力先パスの生成
    expt_path = splta.myfm.setup_expt_directory(os.path.basename(__file__))
    expt_name = os.path.basename(expt_path)
    testset_name = os.path.basename(os.path.dirname(fn2_full))
    output_dir = splta.myfm.setup_output_directory(expt_name, testset_name)
    detected_dir = splta.myfm.setup_output_directory(output_dir, 'detected_mesh')
    intp_detected_dir = splta.myfm.setup_output_directory(output_dir, 'interpolated_mesh')
    interpolation_image_dir = splta.myfm.setup_output_directory(output_dir, 'intermediate')
    dump_detected_dir = splta.myfm.setup_output_directory(output_dir, 'dump_detected_dir')

    # 検出不可能メッシュ
    denied_mesh = list(m_in.is_detectable(len(match), median) for match in mesh_pQ)
    denied_num = len(denied_mesh) - sum(denied_mesh)
    intermediate_testcase_dir = splta.myfm.setup_output_directory(interpolation_image_dir, fn)
    mesh_corners, good_Hs, estimated = m_in.interpolate_mesh(denied_num, Hs, temp_inf, output=True,
                                                        intermediate_dir=intermediate_testcase_dir,
                                                             imgQ=imgQ, imgT=imgT)

    viw = splta.draw_matches_for_meshes(imgQ, imgT, Hs=Hs)
    splta.cv2.imwrite(os.path.join(detected_dir, fn + '.png'), viw)
    splta.cv2.waitKey(1)
    viw = m_in.draw_matches_for_meshes(imgQ, imgT, Hs=good_Hs, vis=None, estimated=estimated)
    splta.cv2.imwrite(os.path.join(intp_detected_dir, 'E_' + fn + '.png'), viw)
    splta.cv2.waitKey(1)
    viw = m_in.draw_matches_for_meshes(imgQ, imgT, Hs=good_Hs, vis=None, estimated=None)
    splta.cv2.imwrite(os.path.join(intp_detected_dir, fn + '.png'), viw)
    splta.cv2.waitKey(1)
    splta.cv2.destroyAllWindows()

    import joblib
    dump_detected_testcase_dir = splta.myfm.setup_output_directory(dump_detected_dir, fn)
    joblib.dump(good_Hs, os.path.join(dump_detected_testcase_dir, 'good_Hs.pikle'), compress=True)
    joblib.dump(mesh_corners, os.path.join(dump_detected_testcase_dir, 'mesh_corners.pikle'), compress=True)
    joblib.dump(estimated, os.path.join(dump_detected_testcase_dir, 'mesh_corners.pikle'), compress=True)

    nodes_has_meshes = temp_inf.get_nodes_has_meshes_id()
    nodes_has_posisions = list(m_in.get_nodes_has_positions(*m, mesh_corners=mesh_corners) for m in nodes_has_meshes)
    nodes_dispersion = list(m_in.get_nodes_dispersion(*vs, imgQ=imgQ) for vs in nodes_has_posisions)
    joblib.dump(nodes_has_posisions, os.path.join(dump_detected_testcase_dir, 'nodes_positions.pikle'), compress=True)
    joblib.dump(nodes_dispersion, os.path.join(dump_detected_testcase_dir, 'nodes_dispersion.pikle'), compress=True)

