# built-in modules
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np

# local modules
from commons.common import Timer
from commons.my_common import load_pikle
from commons.find_obj import init_feature
from commons.affine_base import affine_detect
from commons.template_info import TemplateInfo as TmpInf
from commons.custom_find_obj import explore_match_for_meshes, filter_matches_wcross as c_filter
from commons.custom_find_obj import calclate_Homography, draw_matches_for_meshes
from commons.custom_find_obj import calclate_Homography4splitmesh
from make_database import make_splitmap as mks
from commons import my_file_path_manager as myfm