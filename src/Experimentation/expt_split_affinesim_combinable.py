# python

from make_database import split_affinesim_combinable as splta
from commons.custom_find_obj import explore_match_for_meshes
from commons.my_common import format4pickle_pairs
import os

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
        fn2_full = os.path.abspath(os.path.join(dir_path_full, 'data/inputs/cgs/mltf_qrmarker/000_090-000.png'))
        testset_dir_full = os.path.abspath(os.path.join(dir_path_full, 'data/inputs/cgs/mltf_qrmarker'))
        pr = "mltf_"

    imgQ = splta.cv2.imread(fn1_full, 0)
    imgT = splta.cv2.imread(fn2_full, 0)
    detector, matcher = splta.init_feature(feature_name)

    if imgQ is None:
        print('Failed to load fn1:', fn1_full)
        sys.exit(1)
    print("MARKER: {}".format(fn1_full))

    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)
    print("FEATURE: {}".format(feature_name))

    scols = 8
    srows = 8
    h, w = imgQ.shape[:2]
    template_fn, ext = os.path.splitext(os.path.basename(fn1_full))
    template_information = {"_fn": "tmp.png", "template_img": template_fn,
                            "_cols": w, "_rows": h, "_scols": scols, "_srows": srows, "_nneighbor": 4}
    temp_inf = splta.TmpInf(**template_information)

    try:
        with splta.Timer('Lording pickle'):
            splt_kpQ, splt_descQ = splta.affine_load_into_mesh(template_fn, temp_inf.get_splitnum())
    except ValueError as e:
        print(e.args)
        print('If you need to save {} to file as datavase. ¥n'
              + ' Execute makedb/make_split_combine_featureDB_from_templates.py')
        with splta.Timer('Detection and dividing'):
            splt_kpQ, splt_descQ = splta.affine_detect_into_mesh(detector, temp_inf.get_splitnum(),
                                                                imgQ, simu_param='default')

    m_skQ, m_sdQ, m_k_num, merged_map = splta.combine_mesh_compact(splt_kpQ, splt_descQ, temp_inf)
    list_merged_mesh_id = list(set(splta.np.ravel(merged_map)))
    median = splta.np.nanmedian(m_k_num)

    # 実験用出力先パスの生成
    expt_path = splta.myfm.setup_expt_directory(os.path.basename(__file__))
    expt_name = os.path.basename(expt_path)
    testset_name = os.path.basename(os.path.dirname(fn2_full))
    output_dir = splta.myfm.setup_output_directory(expt_name, testset_name)
    detected_dir = splta.myfm.setup_output_directory(output_dir, 'detected_mesh')
    line_dir = splta.myfm.setup_output_directory(output_dir, 'dmesh_line')
    dump_match_dir = splta.myfm.setup_output_directory(output_dir, 'dump_match_dir')
    dump_detected_dir = splta.myfm.setup_output_directory(output_dir, 'dump_detected_dir')

    fn, ext = os.path.splitext(os.path.basename(fn2_full))
    imgT = splta.cv2.imread(fn2_full, 0)
    if imgT is None:
        print('Failed to load fn2:', fn2_full)
        sys.exit(1)
    print("INPUT: {}".format(fn2_full))

    pool = splta.ThreadPool(processes=splta.cv2.getNumberOfCPUs())
    with splta.Timer('Detection'):
        kpT, descT = splta.affine_detect(detector, imgT, pool=pool, simu_param='test')


    with splta.Timer('matching'):
        mesh_pQ, mesh_pT, mesh_pairs = splta.match_with_cross(matcher, m_sdQ, m_skQ, descT, kpT)

    import joblib
    dump_match_testcase_dir = splta.myfm.setup_output_directory(dump_match_dir, fn)
    print(dump_match_testcase_dir)
    joblib.dump(mesh_pQ, os.path.join(dump_match_testcase_dir, 'mesH_pQ.pikle'), compress=True)
    joblib.dump(mesh_pT, os.path.join(dump_match_testcase_dir, 'mesH_pT.pikle'), compress=True)

    import pickle
    index_mesh_pairs = format4pickle_pairs(mesh_pairs)
    with open(os.path.join(dump_match_testcase_dir, 'mesh_pairs.pickle'), 'wb') as f:
        pickle.dump(index_mesh_pairs, f)
        f.close()
    import make_database.mesh_interpolation as mesti
    _mesh_pQ, _mesh_pT, _mesh_pairs = mesti.load_pickle_match_with_cross(expt_name, testset_name, fn)
    for mQ, mT, mP, _mQ, _mT, _mP in zip(mesh_pQ, mesh_pT, mesh_pairs, _mesh_pQ, _mesh_pT, _mesh_pairs):
        for q, t, p, _q, _t, _p in zip(mQ, mT, mP, _mQ, _mT, _mP):
            if q[0] == _q[0]:
                pass
            else:
                print("qx")
                sys.exit(1)
            if q[1] == _q[1]:
                pass
            else:
                print("qy")
                sys.exit(1)
            if t[0] == _t[0]:
                pass
            else:
                print("tx")
                sys.exit(1)
            if t[1] == _t[1]:
                pass
            else:
                print("ty")
                sys.exit(1)
            if _q[1] == _t[1]:
                print("qとty")
                sys.exit(1)
            if _q[0] == _t[0]:
                print("qとtx")
                sys.exit(1)
            if not p[0].pt[0] == _p[0].pt[0] or not p[0].pt[1] == _p[0].pt[1]:
                print("キーポイント")
                sys.exit(1)
            else:
                pass
            if not p[1].pt[0] == _p[1].pt[0] or not p[1].pt[1] == _p[1].pt[1]:
                print("キーポイント")
                sys.exit(1)
            else:
                pass
    with splta.Timer('estimation'):
        Hs, statuses, pairs = splta.calclate_Homography4splitmesh(mesh_pQ, mesh_pT, mesh_pairs, median=median)
    joblib.dump(Hs, os.path.join(dump_match_testcase_dir, 'Hs.pikle'), compress=True)
    joblib.dump(statuses, os.path.join(dump_match_testcase_dir, 'statuses.pikle'), compress=True)
    index_pairs = tuple(
        tuple((p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in pair) for pair in pairs)
    with open(os.path.join(dump_match_testcase_dir, 'pairs.pickle'), 'wb') as f:
        pickle.dump(index_pairs, f)

    vis = splta.draw_matches_for_meshes(imgQ, imgT, temp_inf=temp_inf,
                                        Hs=Hs, list_merged_mesh_id=list_merged_mesh_id, merged_map=merged_map)
    splta.cv2.imwrite(os.path.join(detected_dir, fn + '.png'), vis)

    viw = explore_match_for_meshes('affine find_obj', imgQ, imgT, pairs, Hs=Hs)

    splta.cv2.imwrite(os.path.join(line_dir, fn + '.png'), viw)
    splta.cv2.destroyAllWindows()

