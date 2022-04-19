
from time import time
from os import makedirs
from os.path import exists, dirname
import shutil
from typing import List
from util.util_colormap import heatmap_to_pseudo_color
from imageio import mimwrite
import numpy as np
import logging
import torch


def log_loss(loss, loss_dict):
    if type(loss) == torch.Tensor:
        loss_string = f'total_loss: {loss.item():.4f}'
    else:
        loss_string = f'total_loss: {loss:.4f}'
    for key, value in loss_dict.items():
        loss_string += f' {key}: {value:.4f}'
    logging.info(loss_string)


def list_of_array_to_img_list(list_of_array, normalize=True):
    max_v = np.max([np.max(arr) for arr in list_of_array])
    min_v = np.min([np.min(arr) for arr in list_of_array])

    def normalize(arr):
        norm_arr = abs(arr-min_v)/abs(max_v-min_v+1e-6)
        norm_arr = np.clip(norm_arr, 0, 1)
        return norm_arr

    def normalize_each(arr):
        output = []
        for i in range(len(arr)):
            min_i = np.min(arr[i])
            max_i = np.max(arr[i])
            norm_arr = abs(arr[i]-min_i)/abs(max_i-min_i+1e-6)
            norm_arr = np.clip(norm_arr, 0, 1)
            output.append(norm_arr)
        return output

    if normalize:
        image_list = [(heatmap_to_pseudo_color(normalize(x))*255).astype(np.uint8)
                      for x in list_of_array]
    else:
        output_x = normalize_each(list_of_array)
        image_list = [(heatmap_to_pseudo_color(x)*255).astype(np.uint8)
                      for x in output_x]

    return image_list


def list_of_array_to_gif(list_of_array, output_path, fps=10):
    max_v = np.max([np.max(arr) for arr in list_of_array])
    min_v = np.min([np.min(arr) for arr in list_of_array])

    def normalize(arr):
        norm_arr = abs(arr-min_v)/abs(max_v-min_v+1e-6)
        norm_arr = np.clip(norm_arr, 0, 1)
        return norm_arr
    image_list = [(heatmap_to_pseudo_color(normalize(x))*255).astype(np.uint8)
                  for x in list_of_array]
    mimwrite(output_path, image_list, fps=fps)


def photo_metric_loss(img_warped, img_gt, occ_mask):
    pass


def get_occ_mask(flow_1_2, flow_2_1, th=1):
    pass


def copy_and_create_dir(src, dst):
    if not exists(dirname(dst)):
        makedirs(dirname(dst))
    shutil.copy(src, dst)


def timethis(prefix):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time()
            result = func(*args, **kwargs)
            end = time()
            logging.info('{} took {:.3f} seconds'.format(prefix, end - start))
            return result
        return wrapper
    return decorator


class Timer(object):
    def __init__(self, annotation_str=''):
        self.annotation = annotation_str

    def __enter__(self):
        self.time = time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        logging.info('{} time: {:.4f}'.format(
            self.annotation, time()-self.time))


class KeyFrameBuffer:
    def __init__(self, buffer_size=5):
        self.buffer = []
        self.buffer_size = buffer_size

    def add_keyframe(self, index):
        if index in self.buffer:
            return
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(index)

    def clear(self):
        del self.buffer
        self.buffer = []


class ResultCache:
    def __init__(self, max_cache_length=100, enable_cpu=False):
        self.max_cache_length = max_cache_length
        self.cuda_cache = {}
        self.cpu_cache = {}
        self.cuda_key_history = []
        self.device = None
        self.enable_cpu = enable_cpu

    def add_item(self, key, value):
        if self.device is None and self.enable_cpu:
            if type(value) == list:
                self.device = value[0].device
            else:
                self.device = value.device
        if key not in self.cpu_cache and self.enable_cpu:
            self.cpu_cache[key] = value.detach().cpu()
        if key not in self.cuda_cache:
            self.cuda_cache[key] = value
            self.cuda_key_history.append(key)
            if len(self.cuda_key_history) > self.max_cache_length:
                key_to_remove = self.cuda_key_history.pop(0)
                del self.cuda_cache[key_to_remove]

    def move_cpu_to_cuda(self, key, value):
        self.cuda_cache[key] = value.to(self.device)
        self.cuda_key_history.append(key)
        if len(self.cache) > self.max_cache_length:
            key_to_remove = self.cuda_key_history.pop(0)
            del self.cuda_cache[key_to_remove]

    def get_item(self, key):
        if key in self.cuda_key_history:
            return self.cuda_cache[key]
        elif key in self.cpu_cache and self.enable_cpu:
            return self.cpu_cache[key].to(self.device)
        else:
            return None

    def empty_cache(self):
        keys = list(self.cuda_cache.keys())
        for k in keys:
            del self.cuda_cache[k]
        del self.cuda_key_history
        self.cuda_cache = {}
        self.cuda_key_history = []
        keys = list(self.cpu_cache.keys())
        for k in keys:
            del self.cpu_cache[k]
        self.cpu_cache = {}
