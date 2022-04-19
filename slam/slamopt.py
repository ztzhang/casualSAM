# %%
import itertools
from PIL.Image import init

from configargparse import init_argument_parser
from slam.util_misc import KeyFrameBuffer, ResultCache, Timer, timethis, list_of_array_to_gif, log_loss
import torch
from slam.losses import cauchy_loss_fn, OccMask, mse_loss_fn, smooth_L1_loss_fn, cauchy_loss_with_uncertainty, get_depth_regularizer

import os
from os.path import join
import numpy as np
from tqdm import tqdm
from networks.goem_opt import WarpImage
# from util.util_3dvideo import save_ply, depth_to_points_Rt, Video_3D_Webpage
from util.util_3dvideo_js import PointCloudVideoWebpage, process_for_pointcloud_video
from glob import glob
import logging
from slam.models.cache import FlowCache
from functools import partial
import torch.nn.functional as F
# %%


class DepthVideoOptimization():
    def __init__(self, opt, depth_video_dataset):
        self.opt = opt
        self.depth_video = depth_video_dataset
        self.keyframe_buffer = KeyFrameBuffer(buffer_size=opt.buffer_size)
        self.depth_prediction_cache = ResultCache(
            max_cache_length=300, enable_cpu=False)
        if opt.loss_fn == 'cauchy':
            self.loss_fn = partial(
                cauchy_loss_with_uncertainty, bias=self.opt.cauchy_bias, self_calibration=self.opt.self_calibration)
        elif opt.loss_fn == 'mse':
            self.loss_fn = mse_loss_fn
        elif opt.loss_fn == 'smooth_L1':
            self.loss_fn = smooth_L1_loss_fn
        else:
            raise NotImplementedError(
                f'loss function {opt.loss_fn} is not supported')
        self.depth_regularizer = get_depth_regularizer(opt.depth_regularizer)
        # self.all_optimizers = [self.depth_video.rotation_optimizer, self.depth_video.translation_optimizer,
        #                        self.depth_video.scale_and_shift_optimizer, self.depth_video.depth_optimizer]
        self.backward_warper = WarpImage()
        self.working_flow_cache = FlowCache(flow_network=self.depth_video.flow_net, occ_warping=self.depth_video.get_valid_flow_mask,
                                            images=self.depth_video.images_orig, output_shape=self.depth_video.opt_shape, flow_batch_size_half=3)

    def to(self, device, cache_data=False):
        self.depth_video.to(device, cache_data=cache_data)
        self.device = device

    def cuda(self, cache_data=False):
        self.to(device='cuda', cache_data=cache_data)
        self.device = 'cuda'

    @property
    def device(self):
        return self.depth_video.device

    def set_all_optimizers(self):
        self.all_optimizers = [self.depth_video.rotation_optimizer, self.depth_video.translation_optimizer,
                               self.depth_video.scale_and_shift_optimizer, self.depth_video.depth_optimizer]
        if self.opt.use_uncertainty:
            self.all_optimizers.append(self.depth_video.uncertainty_optimizer)
        if self.opt.opt_intrinsics:
            self.all_optimizers.append(self.depth_video.intrinsics_optimizer)

    def set_active_optimizers(self, names):
        self.active_optimizers = []
        if 'depth' in names:
            self.active_optimizers.append(self.depth_video.depth_optimizer)
        if 'pose' in names:
            self.active_optimizers.append(self.depth_video.rotation_optimizer)
            self.active_optimizers.append(
                self.depth_video.translation_optimizer)
        if 'scale' in names:
            self.active_optimizers.append(
                self.depth_video.scale_and_shift_optimizer)
        if 'intrinsics' in names and self.opt.opt_intrinsics:
            self.active_optimizers.append(
                self.depth_video.intrinsics_optimizer)
        if 'uncertainty' in names and self.opt.use_uncertainty:
            self.active_optimizers.append(
                self.depth_video.uncertainty_optimizer)

    def init_pair(self, idx_1, idx_2):
        self.all_optimizers = [self.depth_video.pose_optimizer,
                               self.depth_video.depth_optimizer, self.depth_video.scale_and_shift_optimizer]

        self.active_optimizers = [
            self.depth_video.pose_optimizer, self.depth_video.scale_and_shift_optimizer]
        self.keyframe_buffer.add_keyframe(idx_1)
        self.keyframe_buffer.add_keyframe(idx_2)
        with Timer('scale init:'):
            self.optimize_over_keyframe_buffer(
                no_depth_grad=True, iteration=601)
        self.active_optimizers = [
            self.depth_video.pose_optimizer, self.depth_video.depth_optimizer]
        with Timer('full opt:'):
            self.optimize_over_keyframe_buffer(
                no_depth_grad=False, iteration=1001)
        self.active_optimizers = []
        self.keyframe_buffer.clear()

    def optmize_BA_over_window(self, frame_list):
        self.keyframe_buffer.buffer = [x for x in frame_list]
        for f in frame_list:
            self.depth_video.unfreeze_frame(f)
        self.depth_video.freeze_frame(frame_list[0], False)
        self.depth_video.freeze_frame(frame_list[1], False)
        self.active_optimizers = [
            self.depth_video.pose_optimizer, self.depth_video.depth_optimizer]
        with Timer('full opt'):
            self.optimize_over_keyframe_buffer(
                no_depth_grad=False, iteration=2001)
        self.active_optimizers = []

    def local_BA_init_uncertainty(self, frame_list=None, scale_only=False):
        # optimize first pair
        self.set_all_optimizers()
        # self.all_optimizers = [self.depth_video.pose_optimizer,
        #                        self.depth_video.scale_and_shift_optimizer, self.depth_video.depth_optimizer, self.# depth_video.uncertainty_optimizer, self.depth_video.intrinsics_optimizer]
        scale_only_optimizers = ['pose', 'scale', 'uncertainty', 'intrinsics']
        if scale_only:
            fix_scale_optimizers = [
                'pose', 'scale', 'intrinsics', 'uncertainty']
        else:
            fix_scale_optimizers = [
                'pose', 'depth', 'intrinsics', 'uncertainty']

        self.set_active_optimizers(scale_only_optimizers)
        # self.active_optimizers = [
        #     self.depth_video.pose_optimizer, self.depth_video.scale_and_shift_optimizer, self.depth_video.uncertainty_optimizer, self.depth_video.intrinsics_optimizer]
        self.keyframe_buffer.clear()

        if frame_list is None:
            frame_list = list(range(self.depth_video.number_of_frames))
        self.keyframe_buffer.add_keyframe(frame_list[0])
        self.keyframe_buffer.add_keyframe(frame_list[1])
        with Timer('init scale:'):
            self.optimize_over_keyframe_buffer_with_uncertainty(
                no_depth_grad=True, iteration=self.opt.scale_iter)
        # self.set_active_optimizers(fix_scale_optimizers)
        # self.active_optimizers = [
        #     self.depth_video.pose_optimizer, self.depth_video.depth_optimizer, self.depth_video.uncertainty_optimizer, self.depth_video.intrinsics_optimizer]
        if not scale_only:
            with Timer('full opt:'):
                self.optimize_over_keyframe_buffer_with_uncertainty(
                    no_depth_grad=False, iteration=self.opt.joint_iter)
        self.depth_video.freeze_frame(frame_list[0], pose_only=False)
        # self.depth_video.freeze_frame(frame_list[1], scale_only=True)
        logging.info(
            f'scale at frame 0: {self.depth_video.scale_and_shift(frame_list[0])[0].mean().item()}')
        logging.info('first pair initialized')
        for x in range(2, len(frame_list)):
            logging.info(f'---------adding frame {x}-------')
            logging.info(f'adding frame {frame_list[x]}')
            self.keyframe_buffer.add_keyframe(frame_list[x])
            logging.info(f'{self.keyframe_buffer.buffer}')
            self.init_pose(frame_list[x], prev_idx=[
                frame_list[x-1], frame_list[x-1]])  # just use the prvious frame as init
            # scale_data = self.depth_video.scale_and_shift.get_scale_data(
            #     frame_list[x-1])
            # self.depth_video.scale_and_shift.set_scale(
            #     frame_list[x], scale_data)

            # self.set_active_optimizers(scale_only_optimizers)

            # self.active_optimizers = [
            #    self.depth_video.pose_optimizer, self.depth_video.scale_and_shift_optimizer, self.depth_video.uncertainty_optimizer, self.depth_video.intrinsics_optimizer]
            with Timer('scale only opt:'):
                self.optimize_over_keyframe_buffer_with_uncertainty(
                    no_depth_grad=True, iteration=self.opt.scale_iter, last_frame_only=False)
            # self.active_optimizers = [
            #    self.depth_video.pose_optimizer, self.depth_video.depth_optimizer, self.depth_video.uncertainty_optimizer, self.depth_video.intrinsics_optimizer]
            if not scale_only:
                self.set_active_optimizers(fix_scale_optimizers)
                logging.info(
                    f'scale at frame {frame_list[x]}: {self.depth_video.scale_and_shift(frame_list[x])[0].mean().item()}')
                with Timer('full opt:'):
                    self.optimize_over_keyframe_buffer_with_uncertainty(
                        no_depth_grad=False, iteration=self.opt.joint_iter, last_frame_only=False)

            if self.opt.new_init_frozen:
                self.detph_video.freeze_frame(frame_list[x], pose_only=True)
                self.depth_video.freeze_frame(self.keyframe_buffer.buffer[0])
                self.depth_video.freeze_frame(self.keyframe_buffer.buffer[1])
            else:
                # self.depth_video.freeze_frame(frame_list[x], scale_only=True)
                self.depth_video.freeze_frame(
                    self.keyframe_buffer.buffer[0])
            # self.depth_video.freeze_frame(
            #    self.keyframe_buffer.buffer[1], scale_only=True)

            logging.info(
                f'scale at frame {frame_list[x]}: {self.depth_video.scale_and_shift(frame_list[x])[0].mean().item()}')
        self.active_optimizers = []

    def local_BA_init_full(self, frame_list=None, paired=True):
        # optimize first pair
        self.set_all_optimizers()
        scale_only_optimizers = ['pose', 'scale', 'intrinsics']
        fix_scale_optimizers = ['pose', 'depth', 'intrinsics']
        # self.all_optimizers = [self.depth_video.pose_optimizer,
        #                       self.depth_video.scale_and_shift_optimizer, self.depth_video.depth_optimizer]
        # self.active_optimizers = [
        #    self.depth_video.pose_optimizer, self.depth_video.scale_and_shift_optimizer]
        self.set_active_optimizers(scale_only_optimizers)
        self.keyframe_buffer.clear()

        if frame_list is None:
            frame_list = list(range(self.depth_video.number_of_frames))
        self.keyframe_buffer.add_keyframe(frame_list[0])
        self.keyframe_buffer.add_keyframe(frame_list[1])
        with Timer('init scale:'):
            self.optimize_over_keyframe_buffer(
                no_depth_grad=True, iteration=self.opt.scale_iter, last_frame_only=False, paired=paired)
        # self.active_optimizers = [
        #    self.depth_video.pose_optimizer, self.depth_video.depth_optimizer]
        self.set_active_optimizers(fix_scale_optimizers)
        with Timer('full opt:'):
            self.optimize_over_keyframe_buffer(
                no_depth_grad=False, iteration=self.opt.joint_iter, last_frame_only=False, paired=paired)
        self.depth_video.freeze_frame(frame_list[0], scale_only=False)
        self.depth_video.freeze_frame(frame_list[1], scale_only=False)
        logging.info(
            f'scale at frame 0: {self.depth_video.scale_and_shift(frame_list[0])[0].mean().item()}')
        logging.info('first pair initialized')
        for x in range(2, len(frame_list)):
            logging.info(f'---------adding frame {x}-------')
            logging.info(f'adding frame {frame_list[x]}')
            self.keyframe_buffer.add_keyframe(frame_list[x])
            logging.info(f'{self.keyframe_buffer.buffer}')
            self.init_pose(frame_list[x], prev_idx=[
                frame_list[x-1], frame_list[x-1]])  # just use the prvious frame as init
            scale_data = self.depth_video.scale_and_shift.get_scale_data(
                frame_list[x-1])
            self.depth_video.scale_and_shift.set_scale(
                frame_list[x], scale_data)

            # self.active_optimizers = [
            #     self.depth_video.pose_optimizer, self.depth_video.scale_and_shift_optimizer]
            self.set_active_optimizers(scale_only_optimizers)
            with Timer('scale only opt:'):
                self.optimize_over_keyframe_buffer(
                    no_depth_grad=True, iteration=self.opt.scale_iter, last_frame_only=False, paired=paired)

            self.set_active_optimizers(fix_scale_optimizers)
            # self.active_optimizers = [
            #     self.depth_video.pose_optimizer, self.depth_video.depth_optimizer]
            logging.info(
                f'scale at frame {frame_list[x]}: {self.depth_video.scale_and_shift(frame_list[x])[0].mean().item()}')
            with Timer('full opt:'):
                self.optimize_over_keyframe_buffer(
                    no_depth_grad=False, iteration=self.opt.joint_iter, last_frame_only=False, paired=paired)

            self.depth_video.freeze_frame(frame_list[x], scale_only=True)
            self.depth_video.freeze_frame(
                self.keyframe_buffer.buffer[0], scale_only=False)
            self.depth_video.freeze_frame(
                self.keyframe_buffer.buffer[1], scale_only=False)

            logging.info(
                f'scale at frame {frame_list[x]}: {self.depth_video.scale_and_shift(frame_list[x])[0].mean().item()}')
        self.active_optimizers = []

    def init_pose(self, x, prev_idx):
        idx_1, idx_2 = prev_idx
        so3_1, tr_1 = self.depth_video.poses.get_raw_value(idx_1)
        so3_2, tr_2 = self.depth_video.poses.get_raw_value(idx_2)
        self.depth_video.poses.set_rotation_and_translation(
            x, so3_1, 2*tr_1-tr_2)

    def optimize_over_keyframe_buffer_with_uncertainty(self, no_depth_grad=False, iteration=500, last_frame_only=False):
        frame_list = self.keyframe_buffer.buffer
        if no_depth_grad:
            self.optimize_pose_and_scale_over_keyframe_buffer_with_uncertainty(
                frame_list, iteration, last_frame_only)
        else:
            self.BA_over_keyframe_buffer_with_uncertainty(
                frame_list, iteration, last_frame_only)

    def optimize_over_keyframe_buffer(self, no_depth_grad=False, iteration=500, last_frame_only=False, paired=False):
        if not paired:
            raise NotImplementedError
            # if no_depth_grad:
            #     self.optimize_pose_and_scale_over_keyframe_buffer(
            #         iteration=iteration, last_frame_only=last_frame_only)
            # else:
            #     self.BA_over_keyframe_buffer(
            #         iteration=iteration, last_frame_only=last_frame_only)
        else:
            frame_list = self.keyframe_buffer.buffer
            if no_depth_grad:
                self.optimize_pose_and_scale_over_keyframe_buffer_paired(
                    frame_list, iteration, last_frame_only)
            else:
                self.BA_over_keyframe_buffer_paired(
                    frame_list, iteration, last_frame_only)

    def BA_over_all_frames_loss_only(self, frame_list, max_range=None, batch_size=200):
        # here we get all pairs
        all_pairs_candidate = list(itertools.combinations(
            frame_list, 2))
        all_pairs = []
        logging.info('calculating flows')
        with Timer('flow pair generation: '):
            for p in all_pairs_candidate:
                # use cvd style sampling
                gap = abs(p[0]-p[1])
                if max_range is not None:
                    if gap > max_range:
                        continue
                if np.floor(np.log2(gap)) == np.ceil(np.log2(gap)):
                    v_i_j, v_j_i = self.depth_video.get_flow_mask_with_cache(p)
                    if torch.mean(v_i_j.float()).item() > 0.2 and torch.mean(v_j_i.float()).item() > 0.2:
                        all_pairs.append(p)

        bs = 1
        chunks = [all_pairs[x:x+bs] for x in range(0, len(all_pairs), bs)]
        logging.info('sampled {x} pairs'.format(x=len(all_pairs)))
        i_print = self.opt.i_print_full_BA
        with Timer('full BA'):
            with torch.no_grad():
                depth_tensor = self.generate_depth_tensor(frame_list)
                init_depth_tensor = self.get_init_depth_tensor(frame_list)
                loss_list = []
                for chunk in chunks:
                    pred = self.forwrard_prediction_by_pairs(
                        chunk, depth_tensor, depth_tensor_init=init_depth_tensor)
                    loss, loss_dict = self.calculate_loss(
                        pred)*len(chunk)/len(all_pairs)
                    loss.backward(retain_graph=True)
                    loss_list.append(loss.item())
        return loss_list, chunks

    def visualize_flow(self, frame_list, max_range=None):
        all_pairs_candidate = list(itertools.combinations(
            frame_list, 2))
        all_pairs = []
        logging.info('calculating flows')
        with Timer('flow pair generation: '):
            for p in all_pairs_candidate:
                # use cvd style sampling
                gap = abs(p[0]-p[1])
                if max_range is not None:
                    if gap > max_range:
                        continue
                if np.floor(np.log2(gap)) == np.ceil(np.log2(gap)):
                    v_i_j, v_j_i = self.depth_video.get_flow_mask_with_cache(p)
                    if torch.mean(v_i_j.float()).item() > 0.2 and torch.mean(v_j_i.float()).item() > 0.2:
                        all_pairs.append(p)

    def BA_over_all_frames_with_uncertainty(self, frame_list, iteration=500, max_range=None, batch_size=200):
        # here we get all pairs
        self.active_optimizers = [
            self.depth_video.pose_optimizer, self.depth_video.depth_optimizer, self.depth_video.uncertainty_optimizer]
        all_pairs_candidate = list(itertools.combinations(
            frame_list, 2))
        all_pairs = []
        # self.depth_video.freeze_frame(frame_list[0])
        # self.depth_video.freeze_frame(frame_list[1])
        for f in frame_list:
            self.depth_video.unfreeze_frame(f)
        # self.depth_video.freeze_frame(frame_list[0])
        # self.depth_video.freeze_frame(frame_list[1])
        self.depth_video.flow_cache.empty_cache()
        logging.info('calculating flows')
        with Timer('flow pair generation: '):
            for p in all_pairs_candidate:
                # use cvd style sampling
                gap = abs(p[0]-p[1])
                if max_range is not None:
                    if gap > max_range:
                        continue
                if np.floor(np.log2(gap)) == np.ceil(np.log2(gap)):
                    v_i_j, v_j_i = self.depth_video.get_flow_mask_with_cache(p)
                    if torch.mean(v_i_j.float()).item() > 0.2 and torch.mean(v_j_i.float()).item() > 0.2:
                        all_pairs.append(p)

        bs = batch_size
        chunks = [all_pairs[x:x+bs] for x in range(0, len(all_pairs), bs)]
        logging.info('sampled {x} pairs'.format(x=len(all_pairs)))
        i_print = self.opt.i_print_full_BA
        with torch.no_grad():
            scale_tensor = self.get_scale_tensor(frame_list)
        init_depth_tensor = self.get_init_depth_tensor(frame_list)
        with Timer('full BA'):
            for iter in range(iteration):
                for o in self.all_optimizers:
                    o.zero_grad()

                raw_depth_tensor, uncertainty = self.get_raw_depth_tensor_with_uncertainty(
                    frame_list)
                depth_tensor = raw_depth_tensor*scale_tensor

                total_loss = 0

                self.depth_prediction_cache.empty_cache()
                total_loss_dict = {}
                for chunk in chunks[:-1]:
                    # pred = self.forwrard_prediction(i, j)
                    pred = self.forwrard_prediction_by_pairs(
                        chunk, depth_tensor, frame_list, uncertainty_tensor=uncertainty, depth_tensor_init=init_depth_tensor)
                    loss, loss_dict = self.calculate_loss(pred)
                    loss *= len(chunk)/len(all_pairs)
                    loss.backward(retain_graph=True)
                    total_loss += loss.item()
                    if len(total_loss_dict) == 0:
                        for k in loss_dict:
                            total_loss_dict[k] = loss_dict[k] * \
                                len(chunk)/len(all_pairs)
                    else:
                        for k in loss_dict:
                            total_loss_dict[k] += loss_dict[k] * \
                                len(chunk)/len(all_pairs)
                chunk = chunks[-1]
                pred = self.forwrard_prediction_by_pairs(
                    chunk, depth_tensor, frame_list, uncertainty_tensor=uncertainty, depth_tensor_init=init_depth_tensor)
                loss, loss_dict = self.calculate_loss(pred)
                loss *= len(chunk)/len(all_pairs)
                loss.backward()
                total_loss += loss.item()
                if len(total_loss_dict) == 0:
                    for k in loss_dict:
                        total_loss_dict[k] = loss_dict[k] * \
                            len(chunk)/len(all_pairs)
                else:
                    for k in loss_dict:
                        total_loss_dict[k] += loss_dict[k] * \
                            len(chunk)/len(all_pairs)
                if iter % i_print == 0:
                    log_loss(total_loss, total_loss_dict)
                # optimize all relative parameters
                for o in self.active_optimizers:
                    o.step()
                del depth_tensor, uncertainty

    def BA_over_all_frames_low_mem(self, frame_list, iteration=500, max_range=None, batch_size=200, scale_only=False, depth_only=False, pose_only=False, uncertainty_only=False, use_pose_init_reg=False, scale_uncertainty=1):
        # here we get all pairs
        self.set_all_optimizers()
        if scale_only:
            self.set_active_optimizers(
                ['scale', 'pose', 'uncertainty', 'intrinsics'])
        elif depth_only:
            self.set_active_optimizers(['depth'])
        elif pose_only:
            self.set_active_optimizers(['pose'])
        elif uncertainty_only:
            self.set_active_optimizers(['uncertainty'])
        else:
            self.set_active_optimizers(
                ['depth', 'pose', 'uncertainty', 'intrinsics'])
        all_pairs_candidate = list(itertools.combinations(
            frame_list, 2))
        # self.depth_video.freeze_frame(frame_list[0])
        # self.depth_video.freeze_frame(frame_list[1])
        for f in frame_list:
            self.depth_video.unfreeze_frame(f)
        # self.depth_video.freeze_frame(frame_list[0])
        # self.depth_video.freeze_frame(frame_list[1])
        self.working_flow_cache.clear_prebatch()
        logging.info('calculating flows')
        all_pairs = []
        with Timer('flow pair generation: '):
            for p in all_pairs_candidate:
                # use cvd style sampling
                gap = abs(p[0]-p[1])
                if max_range is not None:
                    if gap > max_range:
                        continue
                if np.floor(np.log2(gap)) == np.ceil(np.log2(gap)):
                    all_pairs.append(p)
            self.working_flow_cache.pre_calculate_by_list_of_pairs(
                all_pairs)

        bs = batch_size
        chunks = [all_pairs[x:x+bs] for x in range(0, len(all_pairs), bs)]
        logging.info('sampled {x} pairs'.format(x=len(all_pairs)))
        logging.info('prebatching flows...')
        self.working_flow_cache.prebatch_by_list_of_pairs(chunks)
        i_print = self.opt.i_print_full_BA
        with Timer('full BA'):
            for iter in range(iteration):
                for o in self.all_optimizers:
                    o.zero_grad()
                total_loss = 0
                total_loss_dict = {}
                self.depth_prediction_cache.empty_cache()
                for chunk_id, chunk in enumerate(chunks):
                    # pred = self.forwrard_prediction(i, j)
                    pred = self.forwrard_prediction_by_pairs_low_mem(
                        chunk, chunk_id, with_uncertainty=self.opt.use_uncertainty, scale_uncertainty=scale_uncertainty)
                    loss, loss_dict = self.calculate_loss(pred)
                    loss *= len(chunk)/len(all_pairs)
                    loss.backward()
                    total_loss += loss.item()
                    if len(total_loss_dict) == 0:
                        for k in loss_dict:
                            total_loss_dict[k] = loss_dict[k] * \
                                len(chunk)/len(all_pairs)
                    else:
                        for k in loss_dict:
                            total_loss_dict[k] += loss_dict[k] * \
                                len(chunk)/len(all_pairs)
                if use_pose_init_reg:
                    relative_R, relative_t = self.depth_video.get_current_relative_pose(
                        frame_list)
                    B = relative_R.shape[0]
                    R_loss = F.mse_loss(relative_R.matmul(self.depth_video.relative_R.permute([0, 2, 1])), torch.eye(
                        3).to(self.device)[None, ...].expand(B, -1, -1), size_average=False)/B
                    t_loss = F.mse_loss(
                        relative_t, self.depth_video.relative_t, size_average=False)/B
                    total_loss_dict['relative_R_loss'] = R_loss.item()
                    total_loss_dict['relative_t_loss'] = t_loss.item()
                    pose_loss = (R_loss + t_loss)*self.opt.pose_reg_weight
                    pose_loss.backward()
                if iter % i_print == 0:
                    log_loss(total_loss, total_loss_dict)
                # optimize all relative parameters
                for o in self.active_optimizers:
                    o.step()

    def BA_over_all_frames(self, frame_list, iteration=500, max_range=None, batch_size=200):
        # here we get all pairs
        self.active_optimizers = [
            self.depth_video.pose_optimizer, self.depth_video.depth_optimizer]
        all_pairs_candidate = list(itertools.combinations(
            frame_list, 2))
        all_pairs = []
        # self.depth_video.freeze_frame(frame_list[0])
        # self.depth_video.freeze_frame(frame_list[1])
        for f in frame_list:
            self.depth_video.unfreeze_frame(f)
        # self.depth_video.freeze_frame(frame_list[0])
        # self.depth_video.freeze_frame(frame_list[1])
        logging.info('calculating flows')
        with Timer('flow pair generation: '):
            for p in all_pairs_candidate:
                # use cvd style sampling
                gap = abs(p[0]-p[1])
                if max_range is not None:
                    if gap > max_range:
                        continue
                if np.floor(np.log2(gap)) == np.ceil(np.log2(gap)):
                    v_i_j, v_j_i = self.depth_video.get_flow_mask_with_cache(p)
                    if torch.mean(v_i_j.float()).item() > 0.2 and torch.mean(v_j_i.float()).item() > 0.2:
                        all_pairs.append(p)

        bs = batch_size
        chunks = [all_pairs[x:x+bs] for x in range(0, len(all_pairs), bs)]
        logging.info('sampled {x} pairs'.format(x=len(all_pairs)))
        i_print = self.opt.i_print_full_BA
        init_depth_tensor = self.get_init_depth_tensor(frame_list)
        with Timer('full BA'):
            for iter in range(iteration):
                for o in self.all_optimizers:
                    o.zero_grad()
                depth_tensor = self.generate_depth_tensor(frame_list)

                total_loss = 0
                total_loss_dict = {}
                self.depth_prediction_cache.empty_cache()
                for chunk in chunks[:-1]:
                    # pred = self.forwrard_prediction(i, j)
                    pred = self.forwrard_prediction_by_pairs(
                        chunk, depth_tensor, init_depth_tensor)
                    loss, loss_dict = self.calculate_loss(
                        pred)
                    loss *= len(chunk)/len(all_pairs)
                    loss.backward(retain_graph=True)
                    total_loss += loss.item()
                    if len(total_loss_dict) == 0:
                        for k in loss_dict:
                            total_loss_dict[k] = loss_dict[k] * \
                                len(chunk)/len(all_pairs)
                    else:
                        for k in loss_dict:
                            total_loss_dict[k] += loss_dict[k] * \
                                len(chunk)/len(all_pairs)
                chunk = chunks[-1]
                pred = self.forwrard_prediction_by_pairs(
                    chunk, depth_tensor, init_depth_tensor)
                loss, loss_dict = self.calculate_loss(
                    pred)*len(chunk)/len(all_pairs)
                loss.backward()
                # total_loss.backward()
                total_loss += loss.item()
                if iter % i_print == 0:
                    log_loss(total_loss, total_loss_dict)
                # optimize all relative parameters
                for o in self.active_optimizers:
                    o.step()
                del depth_tensor

    def BA_over_keyframe_buffer_paired(self, frame_list, iteration=500, last_frame_only=False):
        if last_frame_only:
            last_frame = frame_list[-1]
            all_pairs = [[last_frame, x]
                         for x in frame_list[:-1]]
        else:
            all_pairs = list(itertools.combinations(
                frame_list, 2))

        i_print = self.opt.i_print_init_opt
        init_depth_tensor = self.get_init_depth_tensor(frame_list)

        for iter in range(iteration):
            depth_tensor = self.generate_depth_tensor(frame_list)
            # loss = 0
            # self.depth_prediction_cache.empty_cache()
            for i, j in all_pairs:
                pred = self.forwrard_prediction_by_pairs(
                    all_pairs, depth_tensor, frame_list, depth_tensor_init=init_depth_tensor)
                loss, loss_dict = self.calculate_loss(pred)
            if iter % i_print == 0:
                log_loss(loss, loss_dict)
            # optimize all relative parameters
            loss.backward()
            for o in self.active_optimizers:
                o.step()
            for o in self.all_optimizers:
                o.zero_grad()

    def optimize_pose_and_scale_over_keyframe_buffer_paired(self, frame_list, iteration=500, last_frame_only=False):

        # generate all pairs
        if last_frame_only:
            last_frame = frame_list[-1]
            all_pairs = [[last_frame, x]
                         for x in frame_list[:-1]]
        else:
            all_pairs = list(itertools.combinations(
                frame_list, 2))

        # get all depths
        with torch.no_grad():
            raw_depth_tensor = self.get_raw_depth_tensor(frame_list)
            init_raw_depth = self.depth_video.initial_depth[frame_list, ...]

        i_print = self.opt.i_print_init_opt
        # optimization loop
        for iter in range(iteration):
            for o in self.all_optimizers:
                o.zero_grad()
            scale_tensor = self.get_scale_tensor(frame_list)
            depth_tensor = raw_depth_tensor*scale_tensor
            init_depth_tensor = init_raw_depth  # *scale_tensor.detach()
            pred = self.forwrard_prediction_by_pairs(
                all_pairs, depth_tensor, frame_list, depth_tensor_init=init_depth_tensor)
            loss, loss_dict = self.calculate_loss(pred)
            if iter % i_print == 0:
                log_loss(loss, loss_dict)
            # optimize all relative parameters
            loss.backward()
            for o in self.active_optimizers:
                o.step()

    def BA_over_keyframe_buffer_with_uncertainty(self, frame_list, iteration=500, last_frame_only=False):
        if last_frame_only:
            last_frame = frame_list[-1]
            all_pairs = [[last_frame, x]
                         for x in frame_list[:-1]]
        else:
            all_pairs = list(itertools.combinations(
                frame_list, 2))
        i_print = self.opt.i_print_init_opt
        with torch.no_grad():
            scale_tensor = self.get_scale_tensor(frame_list)
            init_depth_tensor = self.get_init_depth_tensor(frame_list)
        for iter in range(iteration):
            raw_depth_tensor, uncertrainty_tensor = self.get_raw_depth_tensor_with_uncertainty(
                frame_list)
            depth_tensor = raw_depth_tensor*scale_tensor
            # loss = 0
            # self.depth_prediction_cache.empty_cache()
            for i, j in all_pairs:
                pred = self.forwrard_prediction_by_pairs(
                    all_pairs, depth_tensor, frame_list, uncertrainty_tensor, depth_tensor_init=init_depth_tensor)
                loss, loss_dict = self.calculate_loss(pred)
            if iter % i_print == 0:
                log_loss(loss, loss_dict)
            # optimize all relative parameters
            loss.backward()
            for o in self.active_optimizers:
                o.step()
            for o in self.all_optimizers:
                o.zero_grad()

    def optimize_pose_and_scale_over_keyframe_buffer_with_uncertainty(self, frame_list, iteration=500, last_frame_only=False):

        # generate all pairs
        if last_frame_only:
            last_frame = frame_list[-1]
            all_pairs = [[last_frame, x]
                         for x in frame_list[:-1]]
        else:
            all_pairs = list(itertools.combinations(
                frame_list, 2))

        # get all depths

        i_print = self.opt.i_print_init_opt
        # optimization loop
        with torch.no_grad():
            raw_depth_tensor = self.get_raw_depth_tensor(frame_list)
            raw_init_depth_tensor = self.depth_video.initial_depth[frame_list, ...]

        for iter in range(iteration):

            for o in self.all_optimizers:
                o.zero_grad()

            scale_tensor = self.get_scale_tensor(frame_list)
            uncertainty_tensor = self.depth_video.predict_uncertainty(
                frame_list)

            depth_tensor = raw_depth_tensor*scale_tensor
            init_depth_tensor = raw_init_depth_tensor*scale_tensor.detach()
            pred = self.forwrard_prediction_by_pairs(
                all_pairs, depth_tensor, frame_list, uncertainty_tensor=uncertainty_tensor, depth_tensor_init=init_depth_tensor)
            loss, loss_dict = self.calculate_loss(pred)
            if iter % i_print == 0:
                log_loss(loss, loss_dict)
            # optimize all relative parameters
            loss.backward()
            for o in self.active_optimizers:
                o.step()

    def generate_depth_tensor(self, frame_list, no_depth_grad=False):
        raw_depth_tensor = self.get_raw_depth_tensor(frame_list)
        scale_tensor = self.get_scale_tensor(frame_list)
        return raw_depth_tensor*scale_tensor

    def get_init_depth_tensor(self, frame_list):
        with torch.no_grad():
            init_depth_tensor = self.depth_video.initial_depth[frame_list, ...]
            scale_tensor = self.get_scale_tensor(frame_list)
        return init_depth_tensor*scale_tensor.detach()

    def get_raw_depth_tensor_with_uncertainty(self, frame_list, no_depth_grad=False):
        depth_tensor, uncertainty_tensor = self.depth_video.predict_depth_with_uncertainty(
            frame_list, no_depth_grad=no_depth_grad)
        return depth_tensor, uncertainty_tensor

    def get_raw_depth_tensor(self, frame_list):
        depth_tensor = self.depth_video.predict_depth(frame_list)
        return depth_tensor

    def get_scale_tensor(self, frame_list):
        scale_tensor = []
        for i in frame_list:
            scale, _ = self.depth_video.scale_and_shift(i)
            scale_tensor.append(scale)
        scale_tensor = torch.cat(scale_tensor, dim=0)
        return scale_tensor

    def get_flow_and_mask_from_list_of_pairs(self, list_of_pairs, cpu_cache=False):
        if cpu_cache is True:
            pass
        else:
            flow_1_2 = []
            flow_2_1 = []
            valid_mask_1 = []
            valid_mask_2 = []
            for pair in list_of_pairs:
                i, j = pair
                flow_1_2.append(
                    self.depth_video.predict_flow_with_cache((i, j)))
                flow_2_1.append(
                    self.depth_video.predict_flow_with_cache((j, i)))
                v1, v2 = self.depth_video.get_flow_mask_with_cache((i, j))
                valid_mask_1.append(v1)
                valid_mask_2.append(v2)
            flow_1_2 = torch.cat(flow_1_2, dim=0)
            flow_2_1 = torch.cat(flow_2_1, dim=0)
            valid_mask_1 = torch.cat(valid_mask_1, dim=0)
            valid_mask_2 = torch.cat(valid_mask_2, dim=0)
            return flow_1_2, flow_2_1, valid_mask_1, valid_mask_2

    def forwrard_prediction_by_pairs_low_mem(self, list_of_pairs, chunk_id, with_uncertainty=False, scale_uncertainty=1):
        data_chunk = self.working_flow_cache.get_flow_and_mask_by_chunk_id(
            chunk_id)
        data_chunk = data_chunk.to(self.device, non_blocking=True)
        # depth tensor is a N1HW tensor of depth maps
        unique_ids = np.unique(list_of_pairs).tolist()
        unique_ids_cuda = torch.LongTensor(unique_ids).to(device=self.device)
        if with_uncertainty:
            depth_tensor, uncertainty_tensor = self.get_raw_depth_tensor_with_uncertainty(
                unique_ids_cuda)
            scale = self.get_scale_tensor(unique_ids)
            depth_tensor = depth_tensor*scale
            uncertainty_tensor = uncertainty_tensor*scale_uncertainty
        else:
            depth_tensor = self.get_raw_depth_tensor(unique_ids)
            scale = self.get_scale_tensor(unique_ids)
            depth_tensor = depth_tensor*scale
            uncertainty_tensor = None
        init_depth_tensor = self.get_init_depth_tensor(unique_ids)
        # init_depth_tensor = init_depth_tensor*scale.detach()

        flow_1_2, flow_2_1, valid_mask_1, valid_mask_2 = self.working_flow_cache.split_chunk(
            data_chunk)

        mask_idx_1 = torch.LongTensor(
            [x[0] for x in list_of_pairs]).to(device=depth_tensor.device)
        mask_idx_2 = torch.LongTensor(
            [x[1] for x in list_of_pairs]).to(device=depth_tensor.device)
        lut_list = unique_ids
        disp_idx_1 = torch.LongTensor(
            [lut_list.index(x[0]) for x in list_of_pairs]).to(device=depth_tensor.device)
        disp_idx_2 = torch.LongTensor(
            [lut_list.index(x[1]) for x in list_of_pairs]).to(device=depth_tensor.device)
        disp_1 = depth_tensor[disp_idx_1, ...]
        disp_2 = depth_tensor[disp_idx_2, ...]
        init_disp_1 = init_depth_tensor[disp_idx_1, ...]
        init_disp_2 = init_depth_tensor[disp_idx_2, ...]
        if uncertainty_tensor is not None:
            uncertainty_1 = uncertainty_tensor[disp_idx_1, ...]
            uncertainty_2 = uncertainty_tensor[disp_idx_2, ...]
        else:
            uncertainty_1 = 0
            uncertainty_2 = 0
        mask_1 = self.depth_video.masks[mask_idx_1, ...]
        mask_2 = self.depth_video.masks[mask_idx_2, ...]
        R_unique, t_unique = self.depth_video.poses(unique_ids)
        # for idx in unique_ids:
        #     R, t = self.depth_video.poses(idx)
        #     R_unique.append(R)
        #     t_unique.append(t)
        # R_unique = torch.cat(R_unique, dim=0)
        # t_unique = torch.cat(t_unique, dim=0)
        R_1_all = R_unique[disp_idx_1, ...]
        t_1_all = t_unique[disp_idx_1, ...]
        R_2_all = R_unique[disp_idx_2, ...]
        t_2_all = t_unique[disp_idx_2, ...]

        # R_1_all = []
        # t_1_all = []
        # R_2_all = []
        # t_2_all = []
        # scale_1_all = []
        # scale_2_all = []
        # for pair in list_of_pairs:
        #     i, j = pair
        #     R_1, t_1 = self.depth_video.poses.forward_index(i)
        #     R_2, t_2 = self.depth_video.poses.forward_index(j)
        #     R_1_all.append(R_1)
        #     R_2_all.append(R_2)
        #     t_1_all.append(t_1)
        #     t_2_all.append(t_2)
        # R_1_all = torch.cat(R_1_all, dim=0)
        # R_2_all = torch.cat(R_2_all, dim=0)
        # t_1_all = torch.cat(t_1_all, dim=0)
        # t_2_all = torch.cat(t_2_all, dim=0)

        K, inv_K = self.get_intrinsics()
        ego_flow_1_2, _ = self.depth_video.depth_based_warping(
            R_1_all, t_1_all, R_2_all, t_2_all, disp_1, K, inv_K)
        ego_flow_2_1, _ = self.depth_video.depth_based_warping(
            R_2_all, t_2_all, R_1_all, t_1_all, disp_2, K, inv_K)
        pred = {}
        # pred['xyz_1'] = self.depth_video.depth_based_warping.unproject_depth(
        #    disp_1, R_1_all, t_1_all, inv_K.detach())
        # pred['xyz_2'] = self.depth_video.depth_based_warping.unproject_depth(
        #     disp_2, R_2_all, t_2_all, inv_K.detach())

        depth_2_from_1, depth_1_from_2 = self.depth_video.depth_based_warping.reproject_depth(
            R_1_all, t_1_all, disp_1, R_2_all, t_2_all, disp_2, K, inv_K)
        pred['depth_2_from_1'] = depth_2_from_1
        pred['depth_1_from_2'] = depth_1_from_2
        pred['disp_1'] = disp_1
        pred['disp_2'] = disp_2
        pred['pred_disp_unique'] = depth_tensor
        pred['init_disp_unique'] = init_depth_tensor
        pred['init_disp_1'] = init_disp_1
        pred['init_disp_2'] = init_disp_2
        pred['ego_flow_1_2'] = ego_flow_1_2
        pred['ego_flow_2_1'] = ego_flow_2_1
        pred['flow_1_2'] = flow_1_2
        pred['flow_2_1'] = flow_2_1
        pred['mask_1'] = mask_1
        pred['mask_2'] = mask_2
        pred['valid_flow_1'] = valid_mask_1
        pred['valid_flow_2'] = valid_mask_2
        pred['uncertainty_1'] = uncertainty_1
        pred['uncertainty_2'] = uncertainty_2
        pred['uncertainty_unique'] = uncertainty_tensor
        return pred

    def get_intrinsics(self):
        if self.opt.opt_intrinsics:
            return self.depth_video.camera_intrinsics.get_K_and_inv()
        else:
            return self.depth_video.K, self.depth_video.inv_K

    def forwrard_prediction_by_pairs(self, list_of_pairs, depth_tensor=None, lut_list=None, uncertainty_tensor=None, depth_tensor_init=None):
        # depth tensor is a N1HW tensor of depth maps
        flow_1_2 = []
        flow_2_1 = []
        valid_mask_1 = []
        valid_mask_2 = []
        for pair in list_of_pairs:
            i, j = pair
            flow_1_2.append(self.depth_video.predict_flow_with_cache((i, j)))
            flow_2_1.append(self.depth_video.predict_flow_with_cache((j, i)))
            v1, v2 = self.depth_video.get_flow_mask_with_cache((i, j))
            valid_mask_1.append(v1)
            valid_mask_2.append(v2)
        flow_1_2 = torch.cat(flow_1_2, dim=0)
        flow_2_1 = torch.cat(flow_2_1, dim=0)
        valid_mask_1 = torch.cat(valid_mask_1, dim=0)
        valid_mask_2 = torch.cat(valid_mask_2, dim=0)

        mask_idx_1 = torch.LongTensor(
            [x[0] for x in list_of_pairs]).to(depth_tensor.device)
        mask_idx_2 = torch.LongTensor(
            [x[1] for x in list_of_pairs]).to(depth_tensor.device)
        if lut_list is not None:
            disp_idx_1 = torch.LongTensor(
                [lut_list.index(x[0]) for x in list_of_pairs]).to(depth_tensor.device)
            disp_idx_2 = torch.LongTensor(
                [lut_list.index(x[1]) for x in list_of_pairs]).to(depth_tensor.device)
        else:
            disp_idx_1 = mask_idx_1
            disp_idx_2 = mask_idx_2
        unique_ids = np.unique(list_of_pairs).tolist()
        if lut_list is not None:
            unique_depth_tensor = depth_tensor
            unique_depth_tensor_init = depth_tensor_init
        else:
            unique_depth_tensor = depth_tensor[unique_ids, ...]
            unique_depth_tensor_init = depth_tensor_init[unique_ids, ...]
            # unique_depth_tensor_init = unique_depth_tensor_init  # * scale.detach()
        disp_1 = depth_tensor[disp_idx_1, ...]
        disp_2 = depth_tensor[disp_idx_2, ...]
        init_disp_1 = depth_tensor_init[disp_idx_1, ...]
        init_disp_2 = depth_tensor_init[disp_idx_2, ...]

        if uncertainty_tensor is not None:
            uncertainty_1 = uncertainty_tensor[disp_idx_1, ...]
            uncertainty_2 = uncertainty_tensor[disp_idx_2, ...]
        else:
            uncertainty_1 = 0
            uncertainty_2 = 0
        if self.opt.cap_uncertainty:
            uncertainty_mask_1 = torch.where(uncertainty_1 > 5, torch.zeros_like(
                uncertainty_1), torch.ones_like(uncertainty_1))
            uncertainty_mask_2 = torch.where(uncertainty_2 > 5, torch.zeros_like(
                uncertainty_2), torch.ones_like(uncertainty_2))
        else:
            uncertainty_mask_1 = 1
            uncertainty_mask_2 = 1
        mask_1 = self.depth_video.masks[mask_idx_1, ...]*uncertainty_mask_1
        mask_2 = self.depth_video.masks[mask_idx_2, ...]*uncertainty_mask_2
        # R_unique = []
        # t_unique = []
        R_unique, t_unique = self.depth_video.poses(unique_ids)
        # for idx in unique_ids:
        #     R, t = self.depth_video.poses(idx)
        #     R_unique.append(R)
        #     t_unique.append(t)
        # R_unique = torch.cat(R_unique, dim=0)
        # t_unique = torch.cat(t_unique, dim=0)
        R_1_all = R_unique[disp_idx_1, ...]
        t_1_all = t_unique[disp_idx_1, ...]
        R_2_all = R_unique[disp_idx_2, ...]
        t_2_all = t_unique[disp_idx_2, ...]

        K, inv_K = self.get_intrinsics()
        ego_flow_1_2, _ = self.depth_video.depth_based_warping(
            R_1_all, t_1_all, R_2_all, t_2_all, disp_1, K, inv_K)
        ego_flow_2_1, _ = self.depth_video.depth_based_warping(
            R_2_all, t_2_all, R_1_all, t_1_all, disp_2, K, inv_K)
        pred = {}
        pred['xyz_1'] = self.depth_video.depth_based_warping.unproject_depth(
            disp_1, R_1_all, t_1_all, inv_K.detach())
        pred['xyz_2'] = self.depth_video.depth_based_warping.unproject_depth(
            disp_2, R_2_all, t_2_all, inv_K.detach())
        depth_2_from_1, depth_1_from_2 = self.depth_video.depth_based_warping.reproject_depth(
            R_1_all, t_1_all, disp_1, R_2_all, t_2_all, disp_2, K, inv_K)
        pred['depth_2_from_1'] = depth_2_from_1
        pred['depth_1_from_2'] = depth_1_from_2
        pred['pred_disp_unique'] = unique_depth_tensor
        pred['init_disp_unique'] = unique_depth_tensor_init
        pred['ego_flow_1_2'] = ego_flow_1_2
        pred['ego_flow_2_1'] = ego_flow_2_1
        pred['flow_1_2'] = flow_1_2
        pred['flow_2_1'] = flow_2_1
        pred['mask_1'] = mask_1
        pred['mask_2'] = mask_2
        pred['valid_flow_1'] = valid_mask_1
        pred['valid_flow_2'] = valid_mask_2
        pred['uncertainty_1'] = uncertainty_1
        pred['uncertainty_2'] = uncertainty_2
        pred['uncertainty_unique'] = uncertainty_tensor
        pred['disp_1'] = disp_1
        pred['disp_2'] = disp_2
        pred['init_disp_1'] = init_disp_1
        pred['init_disp_2'] = init_disp_2
        return pred

    def calculate_loss(self, pred):
        loss_dict = {}
        total_loss = 0
        # xyz_1 = pred['xyz_1']
        # xyz_2 = pred['xyz_2']
        ego_flow_1_2 = pred['ego_flow_1_2']
        ego_flow_2_1 = pred['ego_flow_2_1']
        mask_1 = pred['mask_1']
        mask_2 = pred['mask_2']
        flow_1_2 = pred['flow_1_2']
        flow_2_1 = pred['flow_2_1']
        valid_flow_1 = pred['valid_flow_1']
        valid_flow_2 = pred['valid_flow_2']
        uncertainty_1 = pred['uncertainty_1']
        uncertainty_2 = pred['uncertainty_2']
        if self.opt.uncertainty_channels > 1:
            uncertainty_1_disp = uncertainty_1[:, 1:2, ...]
            uncertainty_2_disp = uncertainty_2[:, 1:2, ...]
            uncertainty_1 = uncertainty_1[:, 0:1, ...]
            uncertainty_2 = uncertainty_2[:, 0:1, ...]

        else:
            uncertainty_1_disp = uncertainty_1
            uncertainty_2_disp = uncertainty_2

        mask_all_1 = mask_1*valid_flow_1  # *uncertainty_mask_1
        mask_all_2 = mask_2*valid_flow_2  # *uncertainty_mask_2

        # calculate scene-flow
        #

        # # supervise uncertainty with flow error
        # flow_err_1 = torch.abs(
        #     flow_1_2 - ego_flow_1_2[:, :2, ...]).sum(1, keepdim=True)
        # flow_err_2 = torch.abs(
        #     flow_1_2 - ego_flow_1_2[:, :2, ...]).sum(1, keepdim=True)

        # uncertainty_loss = torch.mean(
        #     torch.abs(uncertainty_1 - flow_err_1.detach()))
        # uncertainty_loss += torch.mean(torch.abs(uncertainty_2 -
        #                                flow_err_2.detach()))

        # # consistency loss
        # # using xyz coords
        # xyz2_from_xyz1 = self.backward_warper(pred['xyz_1'], flow_2_1)
        # xyz1_from_xyz2 = self.backward_warper(pred['xyz_2'], flow_1_2)

        # depth_consistency_loss = self.loss_fn(
        #     xyz2_from_xyz1, xyz_2.detach(), mask=mask_all_2, c=uncertainty_2.detach())
        # depth_consistency_loss += self.loss_fn(
        #     xyz1_from_xyz2, xyz_1.detach(), mask=mask_all_1, c=uncertainty_1.detach())

        # consistency loss
        # using disparity
        disp_1 = pred['disp_1']
        disp_2 = pred['disp_2']
        # est disparity at camera 1 + cam motion -> disparity value at camera 2 with cam 1 coords
        disp_2_from_1 = 1/(pred['depth_2_from_1']+1e-5)
        # est disparity at camera 2 + cam motion -> disparity value at camera 1 with cam 2 coords
        disp_1_from_2 = 1/(pred['depth_1_from_2']+1e-5)
        # use ego flow here
        # disp_2_from_1 = self.backward_warper(disp_2_from_1, ego_flow_2_1)
        # disp_1_from_2 = self.backward_warper(disp_1_from_2, ego_flow_1_2)
        # disp_2_to_1 = self.backward_warper(disp_2, ego_flow_1_2[:, :2, ...])
        # disp_1_to_2 = self.backward_warper(disp_1, ego_flow_2_1[:, :2, ...])
        # or use optical flow
        if self.opt.use_ego_flow_for_disp_consistency:
            disp_2_to_1 = self.backward_warper(
                disp_2, ego_flow_1_2[:, :2, ...])
            disp_1_to_2 = self.backward_warper(
                disp_1, ego_flow_2_1[:, :2, ...])
        else:
            disp_2_to_1 = self.backward_warper(disp_2, flow_1_2)
            disp_1_to_2 = self.backward_warper(disp_1, flow_2_1)

        # disp_2_from_1 = self.backward_warper(disp_2_from_1, flow_2_1)
        # disp_1_from_2 = self.backward_warper(disp_1_from_2, flow_1_2)
        # uncertainty_loss
        # uncertainty_2_from_1 = self.backward_warper(uncertainty_1, flow_2_1)
        # uncertainty_1_from_2 = self.backward_warper(uncertainty_2, flow_1_2)

        with torch.no_grad():
            uncertainty_consistency_loss = torch.mean(
                pred['uncertainty_unique'][:, 0, ...])

        # uncertainty_consistency_loss = self.loss_fn(
        #     uncertainty_2, torch.maximum(uncertainty_2_from_1, uncertainty_2), mask=mask_all_2)
        # uncertainty_consistency_loss += self.loss_fn(
        #     uncertainty_1, torch.maximum(uncertainty_1_from_2, uncertainty_1), mask=mask_all_1)

        def ordered_ratio(disp_a, disp_b):
            ratio_a = torch.maximum(disp_a, disp_b) / \
                (torch.minimum(disp_a, disp_b)+1e-5)
            return ratio_a - 1

        # depth_consistency_loss = self.loss_fn(
        #     disp_2_from_1, disp_2.detach(), mask=mask_all_2, c=uncertainty_2.detach())
        # depth_consistency_loss += self.loss_fn(
        #     disp_1_from_2, disp_1.detach(), mask=mask_all_1, c=uncertainty_1.detach())

        # should we detach uncertainty here?
        if self.opt.detach_uncertainty_from_depth_consistency:
            depth_consistency_loss = cauchy_loss_with_uncertainty(
                ordered_ratio(disp_2_from_1, disp_2_to_1), 0, mask=mask_all_1, c=uncertainty_1.detach(), bias=self.opt.depth_reg_bias, normalize=True)  # None, bias=1, self_calibration=False)  # uncertainty_1.detach(), bias=0.1, normalize=True)
            depth_consistency_loss += cauchy_loss_with_uncertainty(
                ordered_ratio(disp_1_from_2, disp_1_to_2), 0, mask=mask_all_2, c=uncertainty_2.detach(), bias=self.opt.depth_reg_bias, normalize=True)  # None, bias=1, self_calibration=False)  # uncertainty_2.detach(), bias=0.1, normalize=True)
            # depth_consistency_loss += self.loss_fn(
            #    ordered_ratio(disp_2_from_1, disp_2_to_1_ego), 0, mask=mask_all_1, c=uncertainty_1.detach(), bias=0.1, normalize=True)  # uncertainty_1.detach(), bias=0.1, normalize=True)
            # depth_consistency_loss += self.loss_fn(
            #    ordered_ratio(disp_1_from_2, disp_1_to_2_ego), 0, mask=mask_all_2, c=uncertainty_2.detach(), bias=0.1, normalize=True)  # uncertainty_2.detach(), bias=0.1, normalize=True)
        else:
            depth_consistency_loss = cauchy_loss_with_uncertainty(
                ordered_ratio(disp_2_from_1, disp_2_to_1), 0, mask=mask_all_1, c=uncertainty_1_disp, bias=self.opt.depth_reg_bias, normalize=False)
            depth_consistency_loss += cauchy_loss_with_uncertainty(
                ordered_ratio(disp_1_from_2, disp_1_to_2), 0, mask=mask_all_2, c=uncertainty_2_disp, bias=self.opt.depth_reg_bias, normalize=False)
            # depth_consistency_loss += self.loss_fn(
            #     ordered_ratio(disp_2_from_1, disp_2_to_1_ego), 0, mask=mask_all_1, c=uncertainty_1, bias=0.1, normalize=False)
            # depth_consistency_loss += self.loss_fn(
            #     ordered_ratio(disp_1_from_2, disp_1_to_2_ego), 0, mask=mask_all_2, c=uncertainty_2, bias=0.1, normalize=False)

        if self.opt.use_pixel_weight_for_depth_reg:
            depth_reg_loss = self.depth_regularizer(
                pred['pred_disp_unique'], pred['init_disp_unique'], pixel_wise_weight=pred['uncertainty_unique'].detach().clone()[:, 0:1, ...], pixel_wise_weight_scale=self.opt.pixel_weight_scale, pixel_wise_weight_bias=self.opt.pixel_weight_bias, pixel_weight_normalize=self.opt.pixel_weight_normalization)
        else:
            depth_reg_loss = self.depth_regularizer(
                pred['pred_disp_unique'], pred['init_disp_unique'])

        # depth_reg_loss = self.depth_regularizer(
        #     disp_1, pred['init_disp_1'])  # , uncertainty_1.detach(), pixel_wise_weight_scale=1)
        # depth_reg_loss += self.depth_regularizer(
        #     disp_2, pred['init_disp_2'])  # , uncertainty_2.detach(), pixel_wise_weight_scale=1)

        flow_loss = self.loss_fn(ego_flow_1_2[:, :2, ...],
                                 flow_1_2, mask=mask_all_1, c=uncertainty_1)
        flow_loss = flow_loss + self.loss_fn(ego_flow_2_1[:, :2, ...],
                                             flow_2_1, mask=mask_all_2, c=uncertainty_2)
        total_loss = flow_loss + depth_reg_loss*self.opt.depth_reg_weight + \
            depth_consistency_loss*self.opt.depth_consistency_weight + \
            self.opt.uncertainty_consistency_weight*uncertainty_consistency_loss
        loss_dict['flow_loss'] = flow_loss.item()
        loss_dict['disp_reg'] = depth_reg_loss.item() if type(
            depth_reg_loss) is torch.Tensor else depth_reg_loss
        loss_dict['xyz_cons'] = depth_consistency_loss.item()
        # uncertainty_consistency_loss.item()
        loss_dict['uncertainty_cons'] = uncertainty_consistency_loss.item()
        return total_loss, loss_dict

    def eval_trajectory(self):
        pass

    def save_checkpoint(self, output_path, prefix, frame_list=None):
        logging.info('saving checkpoint...')
        output_path = os.path.join(output_path, prefix)
        num_frames = self.depth_video.number_of_frames
        if frame_list is None:
            frame_list = list(range(num_frames))
        K = self.depth_video.K.cpu().numpy().squeeze()
        os.makedirs(output_path, exist_ok=True)
        with torch.no_grad():
            for n in frame_list:
                disp_pred = self.depth_video.predict_depth(
                    [n]).cpu().numpy().squeeze()
                img = self.depth_video.images[n, ...].cpu().permute(
                    1, 2, 0).numpy()
                R, t = self.depth_video.poses.forward_index(n)
                R = R.cpu().numpy().squeeze()
                t = t.cpu().numpy().squeeze()
                scale, _ = self.depth_video.scale_and_shift(n)
                scale = scale.cpu().numpy().squeeze()
                output_dict = {'disp': disp_pred*scale,
                               'R': R, 't': t, 'K': K, 'img': img}
                np.savez(join(output_path, f'{n:04d}.npz'), **output_dict)
        torch.save(self.depth_video.depth_net.state_dict(),
                   join(output_path, 'depth_net.pth'))
        torch.save(self.depth_video.poses.state_dict(),
                   join(output_path, 'poses.pth'))
        torch.save(self.depth_video.scale_and_shift.state_dict(),
                   join(output_path, 'scale_and_shift.pth'))
        torch.save(self.depth_video.uncertainty_net.state_dict(),
                   join(output_path, 'uncertainty_net.pth'))

    def save_results(self, output_path, frame_list=None, with_uncertainty=False):
        logging.info('saving results.')
        with Timer('save results in'):
            num_frames = self.depth_video.number_of_frames
            if frame_list is None:
                frame_list = list(range(num_frames))
            K = self.get_intrinsics()[0].detach().cpu().numpy().squeeze()
            if hasattr(self.depth_video, 'K'):
                gt_K = self.depth_video.K.cpu().numpy().squeeze()
            else:
                gt_K = K
            os.makedirs(output_path, exist_ok=True)
            with torch.no_grad():
                depth_tensor_list = []
                uncertainty_tensor_list = []
                bs = self.opt.batch_size
                chunks = [frame_list[x:x+bs]
                          for x in range(0, len(frame_list), bs)]
                for chunk in chunks:
                    if not with_uncertainty:
                        depth_tensor = self.depth_video.predict_depth(
                            chunk)
                        uncertainty_tensor = torch.zeros_like(depth_tensor)
                    else:
                        depth_tensor, uncertainty_tensor = self.depth_video.predict_depth_with_uncertainty(
                            chunk)
                    depth_tensor_list.append(depth_tensor)
                    uncertainty_tensor_list.append(uncertainty_tensor)
                depth_tensor = torch.cat(depth_tensor_list, dim=0)
                uncertainty_tensor = torch.cat(uncertainty_tensor_list, dim=0)

                for idn, n in enumerate(frame_list):
                    disp_pred = depth_tensor[idn, ...].cpu().numpy().squeeze()
                    uncertainty_pred = uncertainty_tensor[idn, ...].cpu(
                    ).numpy().squeeze()
                    img = self.depth_video.images[n, ...].cpu().permute(
                        1, 2, 0).numpy()
                    R, t = self.depth_video.poses.forward_index(n)
                    R = R.cpu().numpy().squeeze()
                    t = t.cpu().numpy().squeeze()
                    scale, _ = self.depth_video.scale_and_shift(n)
                    scale = scale.cpu().numpy().squeeze()
                    mask_motion = self.depth_video.get_mask_with_cache(
                        n).cpu().numpy().squeeze()
                    output_dict = {'disp': disp_pred*scale,
                                   'R': R, 't': t, 'K': K, 'img': img, 'mask_motion': mask_motion, 'uncertainty_pred': uncertainty_pred, 'gt_K': gt_K}
                    np.savez(join(output_path, f'{n:04d}.npz'), **output_dict)
            torch.save(self.depth_video.depth_net.state_dict(),
                       join(output_path, 'depth_net.pth'))
            torch.save(self.depth_video.poses.state_dict(),
                       join(output_path, 'poses.pth'))
            torch.save(self.depth_video.scale_and_shift.state_dict(),
                       join(output_path, 'scale_and_shift.pth'))
            torch.save(self.depth_video.camera_intrinsics.state_dict(),
                       join(output_path, 'camera_intrinsics.pth'))
            torch.save(self.depth_video.uncertainty_net.state_dict(),
                       join(output_path, 'uncertainty_net.pth'))
            # all_npzs = sorted(glob(join(output_path, '*.npz')))

            rot_mat = np.asarray([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])

            # rot_mat = np.asarray([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            w = PointCloudVideoWebpage()

            disp_map = []
            uncertainty_map = []

            for idx in frame_list:
                npz_path = join(output_path, f'{idx:04d}.npz')
                npz = np.load(npz_path)
                disp = npz['disp']
                R = npz['R']
                t = npz['t']
                K = npz['K']
                img = npz['img']
                uncertainty_pred = npz['uncertainty_pred']
                mask = np.where(disp < 1e-4, 0, 1)
                mask_motion = self.depth_video.get_mask_with_cache(
                    idx).cpu().numpy().squeeze()
                mask = mask*mask_motion
                pts, rgb, mask, R, t, K, dh, dw = process_for_pointcloud_video(
                    img, disp, mask, R, t, K, rot_mat=rot_mat, downsample=1)
                if not w.camera_registerd:
                    w.register_camera_intrinsics(dh, dw, K, viz_depth=0.05)
                w.add_camera(R, t[:, None], rot_mat)
                w.add_pointcloud(pts, rgb, mask)
                disp_map.append(disp)
                uncertainty_map.append(uncertainty_pred)
            w.save(join(output_path, f'pointcloud_video.html'))
            list_of_array_to_gif(disp_map, join(output_path, 'disp.gif'))
            if len(uncertainty_map[0].shape) > 2:
                C = uncertainty_map[0].shape[0]
                for c in range(C):
                    list_of_array_to_gif([x[c, ...] for x in uncertainty_map], join(
                        output_path, f'uncertainty_c{c}.gif'))
            else:
                list_of_array_to_gif(uncertainty_map, join(
                    output_path, 'uncertainty.gif'))
            del depth_tensor
            del uncertainty_tensor
            del depth_tensor_list
            del uncertainty_tensor_list

    def forwrard_prediction(self, i, j, depths=None, uncertainty_tensor=None):
        raise NotImplementedError
        # pred = {}
        # flow_1_2 = self.depth_video.predict_flow_with_cache((i, j))
        # flow_2_1 = self.depth_video.predict_flow_with_cache((j, i))
        # valid_flow_1, valid_flow_2 = self.depth_video.get_flow_mask_with_cache(
        #     (i, j))
        # if depths is None:
        #     disp_1 = self.depth_prediction_cache.get_item(i)
        #     if disp_1 is None:
        #         disp_1 = self.depth_video.predict_depth([i])
        #         self.depth_prediction_cache.add_item(i, disp_1)
        #     disp_2 = self.depth_prediction_cache.get_item(j)
        #     if disp_2 is None:
        #         disp_2 = self.depth_video.predict_depth([j])
        #         self.depth_prediction_cache.add_item(j, disp_2)
        # else:
        #     disp_1 = depths[0]
        #     disp_2 = depths[1]
        # if uncertainty_tensor is not None:
        #     uncertainty_1 = uncertainty_tensor[0, ...]
        #     uncertainty_2 = uncertainty_tensor[1, ...]
        # else:
        #     uncertainty_1 = 1
        #     uncertainty_2 = 1
        # R_1, t_1 = self.depth_video.poses.forward_index(i)
        # R_2, t_2 = self.depth_video.poses.forward_index(j)
        # scale_1, shift_1 = self.depth_video.scale_and_shift(i)
        # scale_2, shift_2 = self.depth_video.scale_and_shift(j)
        # disp_1_scaled = disp_1*scale_1
        # disp_2_scaled = disp_2*scale_2
        # K, inv_K = self.get_intrinsics()
        # ego_flow_1_2, _ = self.depth_video.depth_based_warping(
        #     R_1, t_1, R_2, t_2, disp_1_scaled, K, inv_K)
        # ego_flow_2_1, _ = self.depth_video.depth_based_warping(
        #     R_2, t_2, R_1, t_1, disp_2_scaled, K, inv_K)
        # pred['ego_flow_1_2'] = ego_flow_1_2
        # pred['ego_flow_2_1'] = ego_flow_2_1
        # pred['flow_1_2'] = flow_1_2
        # pred['flow_2_1'] = flow_2_1
        # pred['mask_1'] = self.depth_video.get_mask_with_cache(i)
        # pred['mask_2'] = self.depth_video.get_mask_with_cache(j)
        # pred['valid_flow_1'] = valid_flow_1
        # pred['valid_flow_2'] = valid_flow_2
        # pred['uncertainty_1'] = uncertainty_1
        # pred['uncertainty_2'] = uncertainty_2
        # return pred

    def optimize_pose_and_scale_over_keyframe_buffer(self, iteration=500, last_frame_only=False, use_cached_depth=False):
        raise NotImplementedError

    # # generate all pairs
    # if last_frame_only:
    #     last_frame = self.keyframe_buffer.buffer[-1]
    #     all_pairs = [[last_frame, x]
    #                  for x in self.keyframe_buffer.buffer[:-1]]
    # else:
    #     all_pairs = list(itertools.combinations(
    #         self.keyframe_buffer.buffer, 2))

    # # get all depths
    # i_print = self.opt.i_print_init_opt

    # if use_cached_depth:

    #     # optimization loop
    #     for iter in range(iteration):
    #         for o in self.all_optimizers:
    #             o.zero_grad()
    #         total_loss = 0
    #         total_loss_dict = {}
    #         for i, j in all_pairs:
    #             pred = self.forwrard_prediction(
    #                 i, j, depths=[self.depth_cache[i:i+1, ...], self.depth_cache[j:j+1, ...]])
    #             loss, loss_dict = self.calculate_loss(pred)
    #             loss = loss / len(all_pairs)
    #             if len(total_loss_dict) == 0:
    #                 for k in loss_dict:
    #                     total_loss_dict[k] = loss_dict[k] / len(all_pairs)
    #             else:
    #                 for k in loss_dict:
    #                     total_loss_dict[k] += loss_dict[k]/len(all_pairs)
    #             total_loss = total_loss+loss
    #         if iter % i_print == 0:
    #             print(total_loss.item())
    #         # optimize all relative parameters
    #         total_loss.backward()
    #         if torch.isnan(loss):
    #             raise Exception('nan loss')
    #         for o in self.active_optimizers:
    #             o.step()
    # else:
    #     with torch.no_grad():
    #         depths = self.depth_video.predict_depth(
    #             self.keyframe_buffer.buffer)

    #     # optimization loop
    #     for iter in range(iteration):
    #         for o in self.all_optimizers:
    #             o.zero_grad()
    #         total_loss = 0
    #         total_loss_dict = {}
    #         for i, j in all_pairs:
    #             pred = self.forwrard_prediction(
    #                 i, j, depths=[self.depth_cache[i:i+1, ...], self.depth_cache[j:j+1, ...]])
    #             loss, loss_dict = self.calculate_loss(pred)
    #             loss = loss / len(all_pairs)
    #             if len(total_loss_dict) == 0:
    #                 for k in loss_dict:
    #                     total_loss_dict[k] = loss_dict[k] / len(all_pairs)
    #             else:
    #                 for k in loss_dict:
    #                     total_loss_dict[k] += loss_dict[k]/len(all_pairs)
    #             total_loss = total_loss+loss
    #         if iter % i_print == 0:
    #             print(total_loss.item())
    #         # optimize all relative parameters
    #         loss.backward()
    #         for o in self.active_optimizers:
    #             o.step()


def BA_over_keyframe_buffer(self, iteration=500, last_frame_only=False):
    raise NotImplementedError
    # if last_frame_only:
    #     last_frame = self.keyframe_buffer.buffer[-1]
    #     all_pairs = [[last_frame, x]
    #                  for x in self.keyframe_buffer.buffer[:-1]]
    # else:
    #     all_pairs = list(itertools.combinations(
    #         self.keyframe_buffer.buffer, 2))

    # i_print = self.opt.i_print_init_opt

    # for iter in range(iteration):
    #     loss = 0
    #     self.depth_prediction_cache.empty_cache()
    #     for i, j in all_pairs:

    #         pred = self.forwrard_prediction(i, j)
    #         loss = loss+self.calculate_loss(pred)/len(all_pairs)
    #     if iter % i_print == 0:
    #         print(loss.item())
    #     # optimize all relative parameters
    #     loss.backward()
    #     for o in self.active_optimizers:
    #         o.step()
    #     for o in self.all_optimizers:
    #         o.zero_grad()


def optmize_pose_and_scale_over_keyframe_buffer(self, iteration=500, last_frame_only=False, use_cached_depth=False):
    raise NotImplementedError

    # # generate all pairs
    # if last_frame_only:
    #     last_frame = self.keyframe_buffer.buffer[-1]
    #     all_pairs = [[last_frame, x]
    #                  for x in self.keyframe_buffer.buffer[:-1]]
    # else:
    #     all_pairs = list(itertools.combinations(
    #         self.keyframe_buffer.buffer, 2))

    # # get all depths
    # i_print = self.opt.i_print_init_opt

    # if use_cached_depth:

    #     # optimization loop
    #     for iter in range(iteration):
    #         for o in self.all_optimizers:
    #             o.zero_grad()
    #         loss = 0
    #         for i, j in all_pairs:
    #             pred = self.forwrard_prediction(
    #                 i, j, depths=[self.depth_cache[i:i+1, ...], self.depth_cache[j:j+1, ...]])
    #             loss = loss+self.calculate_loss(pred)/len(all_pairs)
    #         if iter % i_print == 0:
    #             print(loss.item())
    #         # optimize all relative parameters
    #         loss.backward()
    #         if torch.isnan(loss):
    #             raise Exception('nan loss')
    #         for o in self.active_optimizers:
    #             o.step()
    # else:
    #     with torch.no_grad():
    #         depths = self.depth_video.predict_depth(
    #             self.keyframe_buffer.buffer)

    #     # optimization loop
    #     for iter in range(iteration):
    #         for o in self.all_optimizers:
    #             o.zero_grad()
    #         loss = 0
    #         for i, j in all_pairs:
    #             idi = self.keyframe_buffer.buffer.index(i)
    #             idj = self.keyframe_buffer.buffer.index(j)
    #             pred = self.forwrard_prediction(
    #                 i, j, depths=[depths[idi:idi+1, ...], depths[idj:idj+1, ...]])
    #             loss = loss+self.calculate_loss(pred)/len(all_pairs)
    #         if iter % i_print == 0:
    #             print(loss.item())
    #         # optimize all relative parameters
    #         loss.backward()
    #         for o in self.active_optimizers:
    #             o.step()
