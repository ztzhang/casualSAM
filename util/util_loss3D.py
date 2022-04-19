# %%
import torch
from util.util_img import imread_wrapper
import ujson
import numpy as np
from os.path import join
# %%
# Preproc
# for each object in the image, regress 6 lengths in normalized image coord, and a rotation angle, a bin label wether it is connected to the ground.
# for each each ordered pair, classify their support relationship.
# all image information should be in json.
#scene_label = np.load('/private/home/ztzhang/data/ScanNet/status/all_scene_labels.npz', allow_pickle=True)
train_all_items = ujson.load(open('/private/home/ztzhang/data/ScanNet/status/train_all_data.json'))


# %%
#
max_inst = 0
cnt = 0

for k in train_all_items.keys():
    if 'ground_label' not in train_all_items[k]:
        cnt += 1
    n = len(train_all_items[k]['visible_instance'])
    if n > max_inst:
        key_record = k
    max_inst = max(max_inst, n)
    
    


# %%
scene_id = 'scene0000_00'
frame_num = 2575


def norm(vec3):
    return vec3 / ((vec3[0]**2 + vec3[1]**2 + vec3[2]**2)**0.5)


# Down Up counter_clock_wise: up, left, down, right
# angle
# That's it.
#
item_key = scene_id + '/%05d' % frame_num
# im_path = join('/private/home/ztzhang/data/ScanNet/', scene_id, 'images', '%d.jpg' % frame_num)
# img = imread_wrapper(im_path)
# from matplotlib import pyplot as plt
# plt.imshow(img)
data = train_all_items[item_key]
# %%
from tqdm import tqdm
pbar = tqdm(total=len(train_all_items.keys()))
seg = 10000
cnt = 0
ground_cnt = 0
total = 0
support_pairs_total = 0
support_pairs_pos = 0
valid_ones = []
for k in train_all_items.keys():
    if 'ground_label' not in train_all_items[k]:
        continue
    gl = train_all_items[k]['ground_label']
    pairs = train_all_items[k]['support_matrix']
    pairs = np.array(pairs)
    total += len(gl)
    ground_cnt += sum(gl)
    support_pairs_total += np.prod(pairs.shape)
    support_pairs_pos += np.sum(pairs)
    valid_ones.append(k)
    cnt += 1
    if cnt % seg == 0:
        pbar.update(seg)
#
#
#
#
#
#
#
#
#
#
#
#

# %%
#
#
#
#
#

# %%
# 3D Loss
#
#
#
#
#
#
#
#
#
