
# %%
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn.functional as F

gap = 3
#path = f'/home/zhoutongz_google_com/filestore_fast/zhoutongz/datasets/dataset_local/korean_highline/sequences_select_pairs_midas/IfdO5ujWVwg_78612000/001/shuffle_False_gap_{gap:02d}_sequence_00019.pt'
path = f'/home/zhoutongz_google_com/filestore_fast/zhoutongz/datasets/dataset_local/korean_highline/sequences_select_pairs_midas/jv4m41pbOJE_105940000/001/shuffle_False_gap_{gap:02d}_sequence_00004.pt'

#file:///home/zhoutongz_google_com/filestore/zhoutongz/asset_for_figures/infer_buffer_viz_SM/midas_large_baseline_disp/korean_highline/jv4m41pbOJE_105940000/jv4m41pbOJE_105940000.html#
d = torch.load(path)
# %%
print(d.keys())
plt.imshow(d['mask_2'].squeeze())
# %%
im = Image.fromarray((d['mask_2'].squeeze().numpy().astype(np.uint8)) * 255)
im.save('/home/zhoutongz_google_com/project/smooth_depth_figures/limitation_mask_2.png')
# %%


def backward_flow_warp(im2, flow_1_2):
    H, W, _ = im2.shape
    hh, ww = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
    coord = torch.zeros([1, H, W, 2])
    coord[0, ..., 0] = ww
    coord[0, ..., 1] = hh
    sample_grids = coord + flow_1_2[None, ...]
    sample_grids[..., 0] /= (W - 1) / 2
    sample_grids[..., 1] /= (H - 1) / 2
    sample_grids -= 1
    im = im2.float().permute(2, 0, 1)[None, ...]
    # grids = torch.cat((tw[..., None], th[..., None]), axis=-1).float()[None, ...]
    out = F.grid_sample(im, sample_grids, align_corners=True)
    o = out[0, ...].permute(1, 2, 0)
    return o


# %%
flow_1_2 = d['flow_1_2'].squeeze()
flow_2_1 = d['flow_2_1'].squeeze()
b = backward_flow_warp(flow_2_1, flow_1_2)
error = torch.norm(b + flow_1_2, dim=-1)
#plt.imshow(error * d['mask_2'].squeeze())
# plt.colorbar()
error = error.numpy()

#m = d['mask_2'].squeeze().numpy()
error = np.where(error >= 1, 1, error)

# %%
im = Image.fromarray(((1 - error) * 254).astype(np.uint8))
im.save('/home/zhoutongz_google_com/project/smooth_depth_figures/soft_limitation_mask_2.png')

# %%
m = d['mask_2'].squeeze().numpy()

# %%
