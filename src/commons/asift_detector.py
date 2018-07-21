"""
Feature Detector with Affine Simulation
"""

import cv2
import numpy as np
import logging
# local modules
from commons.custom_find_obj import filter_matches_wcross as c_filter

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


def parameter_generator(longitudes=None, latitudes=None):
    """
    this program is to calculate affine parameter based-on longitude and latitude
    """
    if longitudes is None or latitudes is None:
        return [(1.0, 0.0)]

    arr = [(1.0, 0.0)]
    for t in longitudes:
        for phi in latitudes(t):
            arr.append((t, phi))
    return arr


def calc_affine_params(simu: str ='default'):
    """
    Calculation affine simulation parameter tilt and phi
    You get list object of sets (tilt, phi) as taple
    :param simu: set simulation taype
    :return: list of taple
    """
    longitudes = []
    latitudes = lambda t: []

    if simu == 'default' or simu == 'asift' or simu is None:
        longitudes = 2 ** (0.5 * np.arange(1, 6))
        latitudes = lambda t: np.arange(0, 180, 72.0 / t)

    elif simu == 'degrees':
        """半周する"""
        longitudes = np.reciprocal(np.cos(np.radians(np.arange(10, 90, 10))))
        latitudes = lambda t: np.arange(0, 180, 10)

    elif simu == 'degrees-full':
        """一周する"""
        longitudes = np.reciprocal(np.cos(np.radians(np.arange(10, 90, 10))))
        latitudes = lambda t: np.arange(0, 360, 10)

    elif simu == 'test2':
        # This simulation is Test2 type
        longitudes = np.reciprocal(np.cos(np.radians(np.arange(10, 11, 10))))
        latitudes = lambda t: np.arange(0, 21, 10)

    if simu == 'test' or simu == 'sift':
        #This simulation is Test type"
        return 1.0, 0.0

    return parameter_generator(longitudes, latitudes)

def affine_detect(detector, img, mask=None, pool=None, simu_param='default'):
    """
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transormations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    """

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

    params = calc_affine_params(simu_param)
    keypoints, descrs = [], []
    if pool is None:
        ires = list(map(f, params))
    else:
        ires = pool.imap(f, params)

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