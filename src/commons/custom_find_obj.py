#!/usr/bin/env python

'''
Feature-based image matching sample.

Note, that you will need the https://github.com/opencv/opencv_contrib repo for SIFT and SURF

USAGE
  find_obj.py [--feature=<sift|surf|orb|akaze|brisk>[-flann]] [ <image1> <image2> ]

  --feature  - Feature to use. Can be sift, surf, orb or brisk. Append '-flann'
               to feature name to use Flann-based matcher instead bruteforce.

  Press left mouse button on a feature point to see its matching point.
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2
import numpy as np
import logging
import inspect

from data.ar_template import TemplateMesh as TmpInf
from commons.common import anorm

logger = logging.getLogger(__name__)

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6


def filter_matches_wcross(kp_Q, kp_T, matchesQT, matchesTQ, ratio=0.75):
    """
    filtering matches with cross check
    :param kp_T: Train keypoints
    :param kp_Q: Query keypoints
    :param matchesTQ: Train -> Query dmach list
    :param matchesQT: Query -> Train
    :param ratio: ratio check parameter
    :return: filterd matched points and pairs
    """

    logger.info('Now in {}'.format(inspect.currentframe().f_code.co_name))
    def ratiotest_cross(matches):
        mkpT, mkpQ = [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                mkpT.append(kp_T[m.trainIdx])
                mkpQ.append(kp_Q[m.queryIdx])
        return mkpT, mkpQ

    def crossCheck(dmatchesQT, dmatchesTQ):
        crossOK = []
        for forward in dmatchesQT:
            if len(dmatchesTQ) <= forward[0].trainIdx:
                logger.info('QT.length = %d, TQ.length = %d' % (len(dmatchesQT), len(dmatchesTQ)))
                logger.info('QT.t=%d, TQ.q=%d' % (forward[0].trainIdx, forward[0].queryIdx))
            backward = dmatchesTQ[forward[0].trainIdx]
            if backward[0].trainIdx == forward[0].queryIdx:
                crossOK.append(forward)
        return crossOK

    dmatches_cross = crossCheck(matchesQT, matchesTQ)
    match_kpT, match_kpQ = ratiotest_cross(dmatches_cross)
    pT = np.float32([kp.pt for kp in match_kpT])
    pQ = np.float32([kp.pt for kp in match_kpQ])
    kp_pairsQT = zip(match_kpQ, match_kpT)
    return pQ, pT, list(kp_pairsQT)


def explore_meshes(imgT, Hs=None):
    hT, wT = imgT.shape[:2]
    tmp = TmpInf()

    meshes = []
    for id, H in enumerate(Hs):
        mesh_corners = tmp.calculate_mesh_corners(id)
        if H is not None:
            corners = np.int32(cv2.perspectiveTransform(mesh_corners.reshape(1, -1, 2), H).reshape(-1, 2) + (wT, 0))
            meshes.append(corners)

    return meshes


def draw_matches_for_meshes(imgT, imgQ, Hs=None, vis=None, color=(0, 255, 255)):
    h1, w1 = imgT.shape[:2]
    h2, w2 = imgQ.shape[:2]
    if vis is None:
        vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
        vis[:h1, :w1] = imgT
        vis[:h2, w1:w1 + w2] = imgQ
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    meshes = explore_meshes(imgT, Hs)
    for mesh_corners in meshes:
        cv2.polylines(vis, [mesh_corners], True, color, thickness=7, lineType=cv2.LINE_AA)

    return vis


def explore_match_for_meshes(win, imgT, imgQ, kp_pairs, status=None, Hs=None):
    h1, w1 = imgT.shape[:2]
    h2, w2 = imgQ.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = imgT
    vis[:h2, w1:w1 + w2] = imgQ
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    vis = draw_matches_for_meshes(imgT, imgQ,  Hs, vis)
    vis0 = vis.copy()
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))
    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)

    green = (0, 255, 0)
    red = (0, 0, 255)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv2.line(vis, (x1 - r, y1 - r), (x1 + r, y1 + r), col, thickness)
            cv2.line(vis, (x1 - r, y1 + r), (x1 + r, y1 - r), col, thickness)
            cv2.line(vis, (x2 - r, y2 - r), (x2 + r, y2 + r), col, thickness)
            cv2.line(vis, (x2 - r, y2 + r), (x2 + r, y2 - r), col, thickness)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), green)

    cv2.imshow(win, vis)

    green = (0, 255, 0)
    red = (0, 0, 255)
    kp_color = (51, 103, 236)
    def onmouse(event, x, y, flags, param):
        cur_vis = vis
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cur_vis = vis0.copy()
            r = 8
            m = (anorm(np.array(p1) - (x, y)) < r) | (anorm(np.array(p2) - (x, y)) < r)
            idxs = np.where(m)[0]
            kp1s, kp2s = [], []
            for i in idxs:
                (x1, y1), (x2, y2) = p1[i], p2[i]
                col = (red, green)[status[i]]
                cv2.line(cur_vis, (x1, y1), (x2, y2), col)
                kp1, kp2 = kp_pairs[i]
                kp1s.append(kp1)
                kp2s.append(kp2)
            cur_vis = cv2.drawKeypoints(cur_vis, kp1s, None, flags=4, color=kp_color)
            cur_vis[:, w1:] = cv2.drawKeypoints(cur_vis[:, w1:], kp2s, None, flags=4, color=kp_color)

        cv2.imshow(win, cur_vis)

    cv2.setMouseCallback(win, onmouse)
    return vis


def calclate_Homography(pQ, pT, pairs, hole=0):
    logger.info('Now in {}'.format(inspect.currentframe().f_code.co_name))
    if not hole == 0:
        threshold = hole*0.01
        if threshold < 4:
            threshold = 4
    else:
        threshold = 4
    if len(pQ) >= threshold:
        H, status = cv2.findHomography(pQ, pT, cv2.LMEDS)
        if status is not None and not len(status) == 0:
            logger.info("{0} / {1} = {2:0.3f} inliers/matched=ratio".format(np.sum(status), len(status), np.sum(status)/len(status)))
            # do not draw outliers (there will be a lot of them)
            pairs = [kpp for kpp, flag in zip(pairs, status) if flag]
    else:
        H, status = None, None
        logger.info('%d matches found, not enough for homography estimation' % len(pQ))

    return pairs, H, status

def calclate_Homography_hard(pQ, pT, pairs):
    """calculation homography but hard boarder"""
    logger.info('Now in {}'.format(inspect.currentframe().f_code.co_name))
    if len(pQ) >= 16:
        H, status = cv2.findHomography(pQ, pT, cv2.RANSAC, 5.0)
        logger.info('%d / %d  inliers/matched' % (np.sum(status), len(status)))
        # do not draw outliers (there will be a lot of them)
        pairs = [kpp for kpp, flag in zip(pairs, status) if flag]
    else:
        H, status = None, None
        logger.info('%d matches found, not enough for homography estimation' % len(pQ))

    return pairs, H, status

def calclate_Homography4splitmesh(mesh_pQ, mesh_pT, mesh_pairs, median=0):
    Hs = []
    statuses = []
    kp_pairs_long = []
    for pQ, pT, pairs in zip(mesh_pQ, mesh_pT, mesh_pairs):
        pairs, H, status = calclate_Homography(pQ, pT, pairs, median)
        Hs.append(H)
        if status is not None:
            statuses.extend(status)
        if pairs is not None:
            kp_pairs_long.extend(pairs)
    return Hs, statuses, kp_pairs_long
