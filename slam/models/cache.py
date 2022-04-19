from cv2 import DrawMatchesFlags_DRAW_RICH_KEYPOINTS
import torch
from torch import nn
import torch.nn.functional as F


class FeatListCache():
    """
    This cache is intended to reside on the GPU side for all the time. Depth and Uncertainty uses this cache to avoid heavy recomputation of the backbone features.        
    """

    def __init__(self, number_of_frames=10, list_size=4):
        self.number_of_frames = number_of_frames
        self.feat_list = [[] for _ in range(list_size)]
        self.list_size = list_size
        self.tensor_on_cpu = False

    def add_feat(self, list_of_feat):
        assert len(list_of_feat) == self.list_size
        for idx in range(self.list_size):
            self.feat_list[idx].append(list_of_feat[idx])

    def convert_to_tensor(self):
        self.feat_tensor_list = []
        for idx in range(self.list_size):
            self.feat_tensor_list.append(torch.cat(self.feat_list[idx], dim=0))
        del self.feat_list

    def move_tensor_to_cpu(self):
        self.feat_tensor_list_cpu = []
        for idx in range(self.list_size):
            self.feat_tensor_list_cpu.append(self.feat_tensor_list[idx].cpu(
            ).pin_memory())
        del self.feat_tensor_list
        self.feat_tensor_list = self.feat_tensor_list_cpu
        self.tensor_on_cpu = True

    def __getitem__(self, idx):
        # idx should be a cuda LongTensor
        assert len(idx) >= 1
        if self.tensor_on_cpu:
            idx_cpu = idx.cpu()
            return [x[idx_cpu, ...].to(idx.device, non_blocking=True) for x in self.feat_tensor_list]
        else:
            return [x[idx, ...] for x in self.feat_tensor_list]


class FlowCache():
    """
    all the flows resides on the cpu side, this is intended for the case where #cached flows are large.
    This should also support batching such that there's no need to concatenate for each iteration
    """

    def __init__(self, flow_network, occ_warping, images, output_shape, flow_batch_size_half=2):
        self.flows_by_pair = {}
        self.masks_by_pair = {}
        self.flow_network = flow_network
        self.images = images
        self.bs = flow_batch_size_half
        self.output_shape = output_shape
        self.occ_warping = occ_warping
        output_H, output_W = self.output_shape
        img_H, img_W = self.images[0].shape[-2:]
        self.flow_resize_scale = [output_H/img_H, output_W/img_W]

    def pre_calculate_by_list_of_pairs(self, list_of_pairs):
        # find out pairs not computed yet
        pairs_to_compute = []
        for pair in list_of_pairs:
            if pair not in self.flows_by_pair:
                pairs_to_compute.append(pair)
        invalid_pairs = self.calculate_flow_and_mask(pairs_to_compute)
        for pair in invalid_pairs:
            list_of_pairs.remove(pair)

    @torch.no_grad()
    def calculate_flow_and_mask(self, list_of_pairs):
        # pairs is
        invalid_pairs = []

        chunks = [list_of_pairs[x:x+self.bs]
                  for x in range(0, len(list_of_pairs), self.bs)]
        scale_H, scale_W = self.flow_resize_scale
        for chunk in chunks:
            image_idx_1 = [pair[0] for pair in chunk]
            image_idx_1 += [pair[1] for pair in chunk]
            image_idx_2 = [pair[1] for pair in chunk]
            image_idx_2 += [pair[0] for pair in chunk]
            image_idx_1 = torch.LongTensor(image_idx_1).to(self.images.device)
            image_idx_2 = torch.LongTensor(image_idx_2).to(self.images.device)
            img_1 = self.images[image_idx_1, ...]
            img_2 = self.images[image_idx_2, ...]
            flow = self.flow_network(
                image1=img_1*255, image2=img_2*255, iters=20, test_mode=True)[1]
            # resize flow
            flow = F.interpolate(flow, size=self.output_shape,
                                 mode='bilinear', align_corners=True)
            flow[:, 0, ...] *= scale_W  # x
            flow[:, 1, ...] *= scale_H  # y
            flow_1_2 = flow[:len(chunk), ...]
            flow_2_1 = flow[len(chunk):, ...]
            valid_mask_1 = self.occ_warping(flow_1_2, flow_2_1)
            valid_mask_2 = self.occ_warping(flow_2_1, flow_1_2)
            flow_1_2_cpu = flow_1_2.cpu()
            flow_2_1_cpu = flow_2_1.cpu()
            valid_mask_1_cpu = valid_mask_1.cpu().float()
            valid_mask_2_cpu = valid_mask_2.cpu().float()
            del flow_1_2, flow_2_1, valid_mask_1, valid_mask_2
            for idp, pair in enumerate(chunk):
                if valid_mask_1_cpu[idp, ...].mean() < 0.2 or valid_mask_2_cpu[idp, ...].mean() < 0.2:
                    invalid_pairs.append(pair)
                else:
                    self.masks_by_pair[pair] = [
                        valid_mask_1_cpu[idp:idp+1, ...], valid_mask_2_cpu[idp:idp+1, ...]]
                    self.flows_by_pair[pair] = [
                        flow_1_2_cpu[idp:idp+1, ...], flow_2_1_cpu[idp:idp+1, ...]]
        return invalid_pairs

    def prebatch_by_list_of_pairs(self, list_of_chunks):
        self.data_chunks = []
        for chunk in list_of_chunks:
            data_chunk = []
            for pair in chunk:
                data = []
                flow_1_2, flow_2_1 = self.flows_by_pair[pair]
                valid_mask_1, valid_mask_2 = self.masks_by_pair[pair]
                data.append(flow_1_2)
                data.append(flow_2_1)
                data.append(valid_mask_1)
                data.append(valid_mask_2)
                data = torch.cat(data, dim=1)
                data_chunk.append(data)
            data_chunk = torch.cat(data_chunk, dim=0).pin_memory()
            self.data_chunks.append(data_chunk)

    def get_flow_and_mask_by_chunk_id(self, chunk_id):
        return self.data_chunks[chunk_id]

    def split_chunk(self, data_chunk):
        flow_1_2 = data_chunk[:, :2, ...]
        flow_2_1 = data_chunk[:, 2:4, ...]
        valid_mask_1 = data_chunk[:, 4:5, ...]
        valid_mask_2 = data_chunk[:, 5:6, ...]
        return flow_1_2, flow_2_1, valid_mask_1, valid_mask_2

    def clear_prebatch(self):
        if hasattr(self, 'data_chunks'):
            del self.data_chunks
