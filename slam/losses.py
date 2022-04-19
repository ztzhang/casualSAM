import torch
import torch.nn.functional as F
import math


def depth_consistency_loss():
    pass


def depth_to_global_xyz(disp, K_inv, R, t):
    pass


def get_depth_regularizer(name):
    if name == 'l2':
        return depth_regularization_l2
    elif name == 'stat':
        return depth_regularization_stat
    elif name == 'si':
        return depth_regularization_si
    elif name == 'si_weighted':
        return depth_regularization_si_weighted
    elif name == None or name == 'none':
        return depth_regularization_dummy


def depth_regularization_dummy(depth_pred, depth_init):
    return 0


def depth_regularization_si(depth_pred, depth_init, pixel_wise_weight=None, pixel_wise_weight_scale=1, pixel_wise_weight_bias=1, eps=1e-6, pixel_weight_normalize=False):
    depth_pred = torch.clamp(depth_pred, min=eps)
    depth_init = torch.clamp(depth_init, min=eps)
    log_diff = torch.log(depth_pred)-torch.log(depth_init)
    B, _, H, W = depth_pred.shape
    if pixel_wise_weight is not None:
        if pixel_weight_normalize:
            norm = torch.max(pixel_wise_weight.detach().view(
                B, -1), dim=1, keepdim=False)[0]
            pixel_wise_weight = pixel_wise_weight / \
                (norm[:, None, None, None]+eps)
        log_diff = (pixel_wise_weight*pixel_wise_weight_scale +
                    pixel_wise_weight_bias)*log_diff
    si_loss = torch.sum(
        log_diff**2, dim=[1, 2, 3])/(H*W) - torch.sum(log_diff, dim=[1, 2, 3])**2/((H*W)**2)
    return si_loss.mean()


def depth_regularization_si_weighted(depth_pred, depth_init, pixel_wise_weight=None, pixel_wise_weight_scale=1, pixel_wise_weight_bias=1, eps=1e-6, pixel_weight_normalize=False):
    # scale compute:
    depth_pred = torch.clamp(depth_pred, min=eps)
    depth_init = torch.clamp(depth_init, min=eps)
    log_d_pred = torch.log(depth_pred)
    log_d_init = torch.log(depth_init)
    B, _, H, W = depth_pred.shape
    scale = torch.sum(log_d_init - log_d_pred,
                      dim=[1, 2, 3], keepdim=True)/(H*W)
    if pixel_wise_weight is not None:
        if pixel_weight_normalize:
            norm = torch.max(pixel_wise_weight.detach().view(
                B, -1), dim=1, keepdim=False)[0]
            pixel_wise_weight = pixel_wise_weight / \
                (norm[:, None, None, None]+eps)
        pixel_wise_weight = pixel_wise_weight * \
            pixel_wise_weight_scale + pixel_wise_weight_bias
    else:
        pixel_wise_weight = 1
    si_loss = torch.sum(pixel_wise_weight*(log_d_pred -
                        log_d_init + scale)**2, dim=[1, 2, 3])/(H*W)
    return si_loss.mean()


def depth_regularization_l2(depth_pred, depth_init, pixel_wise_weight=None, pixel_wise_weight_scale=1):
    depth_pred_normalized = depth_pred/(depth_pred.median()+1e-6)
    depth_init_normalized = depth_init/(depth_init.median()+1e-6)
    if pixel_wise_weight is None:
        return torch.mean((depth_pred_normalized - depth_init_normalized)**2)
    else:
        return torch.mean((pixel_wise_weight*pixel_wise_weight_scale+1)*(depth_pred_normalized - depth_init_normalized)**2)


# def depth_regularization_l2(depth_pred, depth_init, pixel_wise_weight=None, pixel_wise_weight_scale=1):
#     B = depth_pred.shape[0]
#     depth_pred_normalized = depth_pred / \
#         (depth_pred.detach().median())  # view(
#     # [B, -1]).median(-1).values[:, None, None, None])
#     depth_init_normalized = depth_init / \
#         (depth_init.detach().median())  # .view(
#     # [B, -1]).median(-1).values[:, None, None, None])
#     if pixel_wise_weight is None:
#         return torch.mean((depth_pred_normalized - depth_init_normalized)**2)
#     else:
#         return torch.mean((pixel_wise_weight*pixel_wise_weight_scale+1)*(depth_pred_normalized - depth_init_normalized)**2)


def depth_regularization_stat(depth_pred, depth_init, crop=1):
    B, C, H, W = depth_pred.shape
    depth_pred_crop = depth_pred.reshape([B, C, crop, H//crop, crop, W//crop])
    depth_init_crop = depth_init.reshape([B, C, crop, H//crop, crop, W//crop])
    mean_pred = torch.mean(depth_pred_crop, [1, 3, 5])
    var_pred = torch.var(depth_pred_crop, [1, 3, 5])
    mean_init = torch.mean(depth_init_crop, [1, 3, 5])
    var_init = torch.var(depth_init_crop, [1, 3, 5])
    return torch.mean((mean_pred-mean_init)**2)+torch.mean((var_pred-var_init)**2)


def rot_loss_fn(rot, rot_gt):
    return torch.sum(
        (rot.detach()@rot_gt.permute([0, 2, 1]) - torch.eye(3, device=rot.device)[None, ...])**2)


def tr_loss_fn(tr, tr_gt):
    return torch.sum((tr-tr_gt)**2)**0.5


def tr_loss_fn_cos(tr, tr_gt, src_t):
    pred_tr = tr.detach()-src_t
    pred_tr = pred_tr/(torch.norm(pred_tr)+1e-6)
    gt_tr = tr_gt-src_t
    gt_tr = gt_tr/(torch.norm(gt_tr)+1e-6)
    return torch.sum(gt_tr*pred_tr)


def disp_loss(disp, disp_gt):
    torch.sum((disp.detach() - disp_gt)**2)**0.5


def mse_loss_fn(estimate, gt, mask):
    v = torch.sum((estimate*mask-gt*mask)**2) / torch.sum(mask)
    return v  # , v.item()


def inv_cauchy(v, c=1):
    return (c**2)*(((torch.exp(v)-1)*2))**0.5

# @torch.jit.script


def cauchy_loss_fn(estimate, gt, mask, c=1.0):
    v = torch.sum(torch.log(0.5*((gt*mask-estimate*mask)/c)**2+1)
                  ) / torch.sum(mask)
    return v  # , (c**2)*(((torch.exp(v.item())-1)*2))**0.5


def cauchy_loss_with_uncertainty(estimate, gt, mask, c=1.0, th=0.0, bias=0.1, self_calibration=False, normalize=False, norm_max=0.4):
    if c is not None and normalize:
        c = norm_max*c/(torch.norm(c, p=2, dim=(1, 2, 3), keepdim=True)+1e-6)
    if c is None:
        c = bias
    err = mask*(gt-estimate)
    err = torch.clamp(torch.abs(err), min=th)
    if self_calibration:
        c = err.detach()
    c = c*mask
    c = bias+c
    v = c+(err**2)/c
    loss = torch.sum(torch.log10(v)-math.log10(bias))/torch.sum(mask)
    return loss


def smooth_L1_loss_fn(estimate, gt, mask, beta=1.0):
    return F.smooth_l1_loss(estimate*mask, gt*mask, beta=beta, reduction='sum') / torch.sum(mask)


def photo_metric_loss(img_1, img_2, flow_1_2, occ_mask_1_2):
    pass


class WarpImage(torch.nn.Module):
    def __init__(self, device):
        super(WarpImage, self).__init__()
        self.base_coord = None

    def init_grid(self, shape, device):
        H, W = shape
        hh, ww = torch.meshgrid(torch.arange(
            H).float(), torch.arange(W).float())
        coord = torch.zeros([1, H, W, 2])
        coord[0, ..., 0] = ww
        coord[0, ..., 1] = hh
        self.base_coord = coord.to(device)
        self.W = W
        self.H = H

    def get_flow_inconsistency_tensor(self, base_coord, img_1, flow_2_1):
        B, C, H, W = flow_2_1.shape
        sample_grids = base_coord + flow_2_1.permute([0, 2, 3, 1])
        sample_grids[..., 0] /= (W - 1) / 2
        sample_grids[..., 1] /= (H - 1) / 2
        sample_grids -= 1
        warped_image_2_from_1 = F.grid_sample(
            img_1, sample_grids, align_corners=True)
        return warped_image_2_from_1

    def forward(self, img_1, flow_2_1):
        B, _, H, W = flow_2_1.shape
        if self.base_coord is None:
            self.init_grid([H, W], device=flow_2_1.device)
        base_coord = self.base_coord.expand([B, -1, -1, -1])


class OccMask(torch.nn.Module):
    def __init__(self, th=3):
        super(OccMask, self).__init__()
        self.th = th
        self.base_coord = None

    def init_grid(self, shape, device):
        H, W = shape
        hh, ww = torch.meshgrid(torch.arange(
            H).float(), torch.arange(W).float())
        coord = torch.zeros([1, H, W, 2])
        coord[0, ..., 0] = ww
        coord[0, ..., 1] = hh
        self.base_coord = coord.to(device)
        self.W = W
        self.H = H

    @torch.no_grad()
    def get_oob_mask(self, base_coord, flow_1_2):
        target_range = base_coord + flow_1_2.permute([0, 2, 3, 1])
        oob_mask = (target_range[..., 0] < 0) | (target_range[..., 0] > self.W-1) | (
            target_range[..., 1] < 0) | (target_range[..., 1] > self.H-1)
        return ~oob_mask[:, None, ...]

    @torch.no_grad()
    def get_flow_inconsistency_tensor(self, base_coord, flow_1_2, flow_2_1):
        B, C, H, W = flow_1_2.shape
        sample_grids = base_coord + flow_1_2.permute([0, 2, 3, 1])
        sample_grids[..., 0] /= (W - 1) / 2
        sample_grids[..., 1] /= (H - 1) / 2
        sample_grids -= 1
        sampled_flow = F.grid_sample(
            flow_2_1, sample_grids, align_corners=True)
        return torch.abs((sampled_flow+flow_1_2).sum(1, keepdim=True))

    def forward(self, flow_1_2, flow_2_1):
        B, _, H, W = flow_1_2.shape
        if self.base_coord is None:
            self.init_grid([H, W], device=flow_1_2.device)
        base_coord = self.base_coord.expand([B, -1, -1, -1])
        oob_mask = self.get_oob_mask(base_coord, flow_1_2)
        flow_inconsistency_tensor = self.get_flow_inconsistency_tensor(
            base_coord, flow_1_2, flow_2_1)
        valid_flow_mask = flow_inconsistency_tensor < self.th
        return valid_flow_mask*oob_mask


def get_occ_mask(flow_1_2, flow_2_1, th=1):
    # flow is in pixel coord, need to convert to grid coord

    pass


'''

def get_oob_mask(flow_1_2):
    H, W, _ = flow_1_2.shape
    hh, ww = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
    coord = torch.zeros([H, W, 2])
    coord[..., 0] = ww
    coord[..., 1] = hh
    target_range = coord + flow_1_2
    m1 = (target_range[..., 0] < 0) + (target_range[..., 0] > W - 1)
    m2 = (target_range[..., 1] < 0) + (target_range[..., 1] > H - 1)
    return (m1 + m2).float().numpy()


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
    im = torch.from_numpy(im2).float().permute(2, 0, 1)[None, ...]
    out = F.grid_sample(im, sample_grids, align_corners=True)
    o = out[0, ...].permute(1, 2, 0).numpy()
    return o

'''
