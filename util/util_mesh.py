import os
if 'DISPLAY' in os.environ:
    del os.environ['DISPLAY']
import sys
from glob import glob
from os.path import join, dirname, exists
import math
import numpy as np
import torch
import sys
sys.path.append('/home/zhoutongz_google_com/project/diffren')
# sys.path.append('/home/zhoutongz_google_com/project/tsdf-fusion-python/')
import camera_utils
import simple_diffren
# import fusion
import trimesh
import moderngl_rasterizer
import matplotlib.pyplot as plt


class Mesh(torch.nn.Module):
    def __init__(self, vtx, face):
        super(Mesh, self).__init__()
        assert len(vtx.shape) == 2
        assert vtx.shape[1] == 3
        # assert vtx.shape[2] == 3
        self.vtx = torch.nn.Parameter(torch.from_numpy(vtx.astype(np.float32))[None, ...])
        faces = torch.from_numpy(face.astype(np.int32))[None, ...]
        self.register_buffer('faces', faces)

    def set_render_param(self, target_width, target_height, num_layers=1):
        self.width = target_width
        self.height = target_height
        self.n_layers = num_layers

    def rasterize(self, world2clip):
        vtx = torch.cat((self.vtx,
                         torch.ones_like(self.vtx[..., 0:1])), dim=-1)
        vtx = torch.matmul(vtx, world2clip.permute([0, 2, 1]))
        barycentric_coords, vertex_ids, mask, triangle_ids = simple_diffren.rasterize_triangles(vtx, self.faces, self.width, self.height, num_layers=1)
        return barycentric_coords, vertex_ids, mask, triangle_ids, vtx

    def render(self, world2clip, vtx_attr, splat=True, shading_func=None, sigma=0.5):
        assert len(vtx_attr.shape) == 3
        assert vtx_attr.shape[0] == 1

        barycentric_coords, vertex_ids, mask, triangle_ids, clip_space_vtx = self.rasterize(world2clip)
        interp_attr = simple_diffren.interpolate_vertex_attribute(vtx_attr,
                                                                  barycentric_coords, vertex_ids, mask)
        B, H, W, L, C = interp_attr.shape
        if shading_func is not None:
            shaded_buffer = shading_func(interp_attr.reshape([-1, C]))
        else:
            shaded_buffer = interp_attr
        shaded_buffer = shaded_buffer.reshape([B, H, W, L, -1])  # .squeeze(3)

        if splat:
            clip_space_buffer = simple_diffren.interpolate_vertex_attribute(clip_space_vtx.detach(), barycentric_coords, vertex_ids, mask)

            ndc_xyz = clip_space_buffer[..., :3] / (clip_space_buffer[..., 3:4] + 1e-6)
            viewport_xyz = (ndc_xyz + 1.0) * torch.tensor(
                [self.width, self.height, 1], dtype=torch.float32).cuda().reshape([1, 1, 1, 1, 3]) * 0.5
            #import pdb
            # pdb.set_trace()
            shaded_buffer_spalt, _, _ = simple_diffren.splat_at_pixel_centers(viewport_xyz, torch.cat([shaded_buffer, mask[..., None, :].float()], -1), sigma)
            return shaded_buffer_spalt
        return shaded_buffer


class VtxAttr(torch.nn.Module):
    def __init__(self, n_vtx, n_ch, uniform_init=True):
        super().__init__()
        self.vtx_attr = torch.nn.Parameter(torch.zeros([1, n_vtx, n_ch]))
        if uniform_init:
            self.vtx_attr.data.uniform_(-1, 1)

    def forward(self):
        return self.vtx_attr
