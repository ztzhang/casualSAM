import numpy as np
from glob import glob
from os.path import join
from argparse import Namespace
import torch
import sys
import numpy as np
import torch
from PIL import Image
from glob import glob
from os.path import join
import cv2
from configs import sintel_data_pattern, sintel_seg_pattern
import os

TAG_FLOAT = 202021.25

TAG_CHAR = 'PIEH'


def resize_flow(flow, size):
    resized_width, resized_height = size
    H, W = flow.shape[:2]
    scale = np.array((resized_width / float(W), resized_height / float(H))).reshape(
        1, 1, -1
    )
    resized = cv2.resize(
        flow, dsize=(
            resized_width, resized_height), interpolation=cv2.INTER_CUBIC
    )
    resized *= scale
    return resized


def get_path(track_name='alley_2'):
    if os.path.exists('/mnt/ssd_1/sintel'):
        data_pattern = '/mnt/ssd_1/sintel/{{mode}}/{track_name}'
    else:
        data_pattern = sintel_data_pattern
    sintel_path = data_pattern.format(
        track_name=track_name)
    sintel_cam_path = sintel_path.format(mode='camdata_left')
    sintel_depth_path = sintel_path.format(mode='depth')
    sintel_flow_path = sintel_path.format(mode='flow')
    sintel_seg_path = sintel_seg_pattern.format(track_name=track_name)
    sintel_img_path = sintel_path.format(mode='clean_left')
    number_of_frames = len(sorted(glob(join(sintel_img_path, '*'))))
    return sintel_cam_path, sintel_depth_path, sintel_flow_path, sintel_seg_path, sintel_img_path, number_of_frames


def img_read(filename):
    im = Image.open(filename)
    return np.asarray(im)/255


def seg_read(filename):
    im = Image.open(filename)
    seg = np.asarray(im).astype(float)/255.
    if len(seg.shape) == 3:
        return seg[..., 0]
    else:
        return seg


def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename, 'rb')
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(
        TAG_FLOAT, check)
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(
        width, height)
    depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def cam_read(filename):
    """ Read camera data, return (M,N) tuple.

    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename, 'rb')
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(
        TAG_FLOAT, check)
    M = np.fromfile(f, dtype='float64', count=9).reshape((3, 3))
    N = np.fromfile(f, dtype='float64', count=12).reshape((3, 4))
    return M, N


def flow_read(filename):
    """ Read optical flow from file, return (U,V) tuple.

    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    f = open(filename, 'rb')
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert check == TAG_FLOAT, ' flow_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(
        TAG_FLOAT, check)
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' flow_read:: Wrong input size (width = {0}, height = {1}).'.format(
        width, height)
    tmp = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width*2))
    u = tmp[:, np.arange(width)*2]
    v = tmp[:, np.arange(width)*2 + 1]
    return u, v


def process_sintel_cam(cam1):

    K = cam1[0]
    src_R = cam1[1][:3, :3]
    src_t = cam1[1][:3, 3:]

    # convert to c2w
    src_R = src_R.T
    src_t = -src_R@src_t

    return K, src_R, src_t
