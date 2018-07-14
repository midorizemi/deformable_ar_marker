"""
Feature Detector with Affine Simulation
"""

import cv2
import numpy as np
import logging
import detector.affine_simulation_parameters as asparams
from detector.affine_simulation_parameters import asift_basic_parameter_gene as asift_prams

def affine_skew(tilt, phi, img, mask=None):
    """
    affine_skew(tilt, phi, img, mask=None) calculates skew_img, skew_mask, affine_inverse
    affine_inverse - is an affine transform matrix from skew_img to img
    """
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    affine = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        affine = np.float32([[c, -s], [s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32(np.dot(corners, affine.T))
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1, -1, 2))
        affine = np.hstack([affine, [[-x], [-y]]])
        img = cv2.warpAffine(img, affine, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        affine[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, affine, (w, h), flags=cv2.INTER_NEAREST)
    aff_inverse = cv2.invertAffineTransform(affine)
    return img, mask, aff_inverse

def a_detect(p, detector, img):
    """
    Calculate Features with Affine Skewing
    """
    t, phi = p
    timg, tmask, Ai = affine_skew(t, phi, img)
    keypoints, descrs = detector.detectAndCompute(timg, tmask)
    for kp in keypoints:
        x, y = kp.pt
        kp.pt = tuple(np.dot(Ai, (x, y, 1)))
    if descrs is None:
        descrs = []
    return keypoints, descrs

def w_a_detect(args):
    a_detect(*args)

def calc_affine_params(simu: str ='default'):
    """
    Calculation affine simulation parameter tilt and phi
    You get list object of sets (tilt, phi) as taple
    :param simu: set simulation taype
    :return: list of taple
    """
    if simu == 'default' or simu == 'asift' or simu is None:
        return asparams.asift_basic_parameter_gene()

    if simu == 'degrees':
        """半周する"""
        return asparams.ap_hemisphere_gene(max_t=90, offset_t=10, max_p=180, offset_p=10)

    if simu == 'degrees-full':
        """一周する"""
        return asparams.ap_hemisphere_gene(max_t=90, offset_t=10, max_p=360, offset_p=10)

    if simu == 'test2':
        # This simulation is Test2 type
        return asparams.ap_hemisphere_gene(max_t=11, offset_t=10, max_p=21, offset_p=10)

    if simu == 'test' or simu == 'sift':
        #This simulation is Test type"
        return 1.0, 0.0



def affine_detect(detector, img, mask=None, pool=None, simu_param=asift_prams()):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transormations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img, mask)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))
        if descrs is None:
            descrs = []
        return keypoints, descrs

    keypoints, descrs = [], []
    if pool is None:
        ires = list(map(f, simu_param))
    else:
        ires = pool.imap(f, simu_param)

    for i, (k, d) in enumerate(ires):
        print('affine sampling: {0:d} / {1:d}\r'.format(i+1, len(params)), end='')
        keypoints.extend(k)
        descrs.extend(d)

    return keypoints, np.array(descrs)

def match_with_cross(matcher, descQ, kpQ, descT, kpT):
    raw_matchesQT = matcher.knnMatch(descQ, trainDescriptors=descT, k=2)
    raw_matchesTQ = matcher.knnMatch(descT, trainDescriptors=descQ, k=2)
    pQ, pT, pairs = c_filter(kpQ, kpT, raw_matchesQT, raw_matchesTQ)
    return pQ, pT, pairs