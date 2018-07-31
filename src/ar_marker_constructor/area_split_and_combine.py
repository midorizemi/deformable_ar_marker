# built-in modules
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np

# local modules
from data.ar_template import TemplateMesh
from commons.asift_detector import load_from_pickle as load_pickle, affine_detect
from commons.debug_tools import set_trace
from commons.custom_find_obj import filter_matches_wcross as c_filter


def split_kd(keypoints, descrs, tempmesh: TemplateMesh):
    assert isinstance(keypoints, list)
    assert isinstance(descrs, np.ndarray)
    split_tmp_img = tempmesh.make_polygon_color_map()

    descrs_list = descrs.tolist()
    splits_k = [[] for row in range(tempmesh.h_split)]
    splits_d = [[] for row in range(tempmesh.h_split)]

    for keypoint, descr in zip(keypoints, descrs_list):
        x, y = np.int32(keypoint.pt)
        if x < 0 or x >= 800:
            if x < 0:
                x = 0
            else:
                x = 799
        if y < 0 or y >= 600:
            if y < 0:
                y = 0
            else:
                y = 599
        splits_k[split_tmp_img[y, x][0]].append(keypoint)
        splits_d[split_tmp_img[y, x][0]].append(descr)

    for i, split_d in enumerate(splits_d):
        splits_d[i] = np.array(split_d, dtype=np.float32)

    return splits_k, splits_d


def combine_mesh(splt_k, splt_d, temp_inf):
    """
    分割したメッシュをキーポイント分布にしたがってマージする
    :type temp_inf: TmpInf
    :param splt_k:
    :param splt_d:
    :param temp_inf:
    :return:
    """
    mesh_map = temp_inf.get_mesh_map()
    mesh_k_num = np.array([len(keypoints) for keypoints in splt_k]).reshape(temp_inf.get_mesh_shape())

    for i, kd in enumerate(zip(splt_k, splt_d)):
        """
        矩形メッシュをマージする．4近傍のマージ．順番は左，上，右，下．
        最大値のところとマージする
        :return:
        """
        meshid_list = temp_inf.get_meshidlist_nneighbor(i)
        self_id = mesh_map[temp_inf.get_meshid_index(i)]
        self_k_num = mesh_k_num[temp_inf.get_meshid_index(self_id)]
        if not self_id == i or len(np.where(mesh_map == i)[0]) > 1 or self_k_num == 0:
            """すでにマージされている"""
            continue
        # 最大値のindexを求める．
        dtype = [('muki', int), ('keypoint_num', int), ('merge_id', int)]
        # tmp = np.array([list(len(meshid_list)-index, mesh_k_num[temp_inf.get_meshid_vertex(id)], id )
        #                for index, id in enumerate(meshid_list) if id is not None]).astype(np.int64)
        tmp = []
        # for index, id in enumerate(meshid_list):
        #     if id is not None:
        #         tmp.extend([len(meshid_list) - index, mesh_k_num[temp_inf.get_meshid_vertex(id)], id])
        # tmp = np.array(tmp)reshape(int(len(tmp)/3, 3).astype(np.int64)
        try:
            for index, id in enumerate(meshid_list):
                if id is not None:
                    tmp.append([len(meshid_list) - index, mesh_k_num[temp_inf.get_meshid_index(id)], id])
        except(IndexError):
            set_trace()

        tmp = np.array(tmp).astype(np.int64)
        median_nearest = np.median(tmp[:, 1])
        if median_nearest < self_k_num:
            # TODO マージ判定
            # 近傍中の中央値よりも注目メッシュのキーポイント数が大きい場合は無視する
            continue
        tmp.dtype = dtype
        tmp.sort(order=['keypoint_num', 'muki'])  # 左回りでかつキーポイント数が最大
        # idにself_idをマージする, 昇順なので末端
        merge_id = tmp[-1][0][2]
        mesh_map[temp_inf.get_meshid_index(i)] = merge_id
        splt_k[merge_id].extend(kd[0])
        mesh_k_num[temp_inf.get_meshid_index(merge_id)] = mesh_k_num[temp_inf.get_meshid_index(merge_id)] + self_k_num
        try:
            np.concatenate((splt_d[merge_id], kd[1]))
        except(IndexError, ValueError):
            set_trace()

        # マージされて要らなくなったメッシュは消す
        splt_k[i] = None
        splt_d[i] = None
        mesh_k_num[temp_inf.get_meshid_index(self_id)] = 0

    return splt_k, splt_d, mesh_k_num, mesh_map


def combine_mesh_compact(splt_k, splt_d, temp_inf):
    sk, sd, mesh_k_num, merged_map = combine_mesh(splt_k, splt_d, temp_inf)
    m_sk = compact_merged_splt(sk)
    m_sd = compact_merged_splt(sd)
    return m_sk, m_sd, mesh_k_num, merged_map


def compact_merged_splt(m_s):
    return [x for x in m_s if x is not None]


def match_with_cross(matcher, meshList_descQ, meshList_kpQ, descT, kpT):
    meshList_pQ = []
    meshList_pT = []
    meshList_pairs = []
    for mesh_kpQ, mesh_descQ in zip(meshList_kpQ, meshList_descQ):
        raw_matchesQT = matcher.knnMatch(mesh_descQ, trainDescriptors=descT, k=2)
        raw_matchesTQ = matcher.knnMatch(descT, trainDescriptors=mesh_descQ, k=2)
        pQ, pT, pairs = c_filter(mesh_kpQ, kpT, raw_matchesQT, raw_matchesTQ)
        meshList_pT.append(pT)
        meshList_pQ.append(pQ)
        meshList_pairs.append(pairs)
    return meshList_pQ, meshList_pT, meshList_pairs


def count_keypoints(splt_kpQ):
    len_s_kp = 0
    for kps in splt_kpQ:
        len_s_kp += len(kps)
    return len_s_kp


def affine_load_into_mesh(temp_mesh: TemplateMesh, pickle_name):
    import os
    pickle_path = temp_mesh.data_path.get_feature_pickle(pickle_name)
    if not os.path.exists(pickle_path):
        print('Not found {}'.format(pickle_path))
        raise ValueError('Failed to load pikle:', pickle_path)
    kp, des = load_pickle(pickle_path)
    return split_kd(kp, des, temp_mesh)


def detect_into_mesh(detector, temp_mesh: TemplateMesh, img1, mask=None, simu_param='default'):
    pool = ThreadPool(processes=cv2.getNumberOfCPUs())
    kp, desc = affine_detect(detector, img1, mask, pool=pool, simu_param=simu_param)
    return split_kd(kp, desc, temp_mesh)
