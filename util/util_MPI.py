# %%
import numpy as np
import torch


# %%

def normalize_uv_intersections(uv, uv_boundingboxes, terminal_val=np.pi):
    """convert uv coord to global coord

    Args:
        intersection (FloatTensor): uv coord, [NxKx2], K being number of planes.
        uv_boundingboxes (List of Float32): 1xKx[sh, sw, ch, cw]
    """
    uv[..., 0] -= uv_boundingboxes[:, :, 3]  # cw
    # * resolution_tensor[None, :, 1]  # sw
    uv[..., 0] /= uv_boundingboxes[:, :, 1] * 0.5
    uv[..., 0] -= 1
    uv[..., 0] *= 2 * terminal_val
    uv[..., 1] -= uv_boundingboxes[:, :, 2]  # ch
    uv[..., 1] /= uv_boundingboxes[:, :, 0] * 0.5
    uv[..., 1] -= 1
    uv[..., 1] *= 2 * terminal_val
    return uv

# def normalize_intersections(uv, box, v=torch.pi):
#    ul_u, ul_v, br
#    uv[..., 0] -= box[None,:, 3]  # cw # this is


def uv2coord_batch(uv, coord_to_uv_tensor, resolution_tensor):
    # sh, sw, ch, cw = sc
    uv[..., 0] -= coord_to_uv_tensor[None, :, 3]  # cw
    uv[..., 0] /= coord_to_uv_tensor[None, :, 1] * \
        0.5 * resolution_tensor[None, :, 1]  # sw
    uv[..., 0] -= 1
    uv[..., 1] -= coord_to_uv_tensor[None, :, 2]  # ch
    uv[..., 1] /= coord_to_uv_tensor[None, :, 0] * \
        0.5 * resolution_tensor[None, :, 0]  # sh
    uv[..., 1] -= 1


def get_crop_coord(shape, n_crop, crop_idh, crop_idw):
    H, W = shape
    H_indexes = np.linspace(0, H, num=n_crop + 1, endpoint=True, dtype=int)
    W_indexes = np.linspace(0, W, num=n_crop + 1, endpoint=True, dtype=int)
    H_beg, H_end = H_indexes[[crop_idh, crop_idh + 1]]
    W_beg, W_end = W_indexes[[crop_idw, crop_idw + 1]]
    linear_idx_all = torch.arange(H * W).reshape([H, W])
    linear_idx = linear_idx_all[H_beg: H_end, W_beg: W_end]
    linear_idx = torch.reshape(linear_idx, [-1])
    return H_beg, H_end, W_beg, W_end, linear_idx
# %%


def depth_to_points(depth, cam_c2w, K):
    H, W = depth.shape
    coord = np.ones([H, W, 3])
    xx, yy = np.meshgrid(np.arange(W).astype(float),
                         np.arange(H).astype(float))
    coord[..., 0] = xx
    coord[..., 1] = yy
    R = cam_c2w[: 3, : 3].T
    t = cam_c2w[: 3, 3: 4].T
    depth = depth[..., None]
    pt_global = (depth * (coord @ np.linalg.inv(K).T)) @ R + t
    return pt_global.reshape([-1, 3])


def planes_from_cam(pose, depth_samples):

    planes = []
    for d in depth_samples:
        sample_oxy = np.asarray([[0., 0., d, 1.],
                                 [1., 0., d, 1.],
                                 [0., 1., d, 1.]])
        # sample_x = np.array([1., 0., d, 1.])
        # sample_y = np.array([0., 1., d, 1.])
        global_oxy = sample_oxy @ (pose.T)
        global_xy = global_oxy[1:, : -1] - global_oxy[0: 1, : -1]
        global_o = global_oxy[0, : -1]
        # print(global_oxy)
        normal_dir = np.cross(global_xy[0, :], global_xy[1, :])
        d = -global_o.dot(normal_dir)
        # print(normal_dir)
        # print(global_xy)
        # print(d)

        planes.append(np.concatenate(
            [normal_dir, global_xy[0, :], global_xy[1, :], [d]], 0))
    return np.stack(planes, 0)


def ray_plane_intersect(ro, rd, planes, eps=1e-8):
    """do ray plane intersection

    Args:
        ro (FloatTensor): Ray origin, [Nx3]
        rd (FloatTensor): Ray direction, unit length [Nx3]
        planes (list of FloatTensor):  Plane params [K, 10], where a = planes[:,:3], b = planes[:,9] and planes are defined by aTx+b=0. Planes[:,3:6] being plane x directions, Planes[:,6:9] being y directions.
    Returns:
        intersections (FloatTensor):  [Nxkx2], intersecting points in plane paramterization.
        intersection_global (FloatTensor):  [Nxkx3], intersecting points in global coordinates.
        t (FloatTensor) [NxKx1] depth for each intersection
    """

    plane_norm = planes[:, : 3]
    plane_offset = planes[:, 9:]
    t = -plane_offset[None, :, :] - \
        torch.matmul(plane_norm[None, :, None, :],
                     ro[:, None, :, None])[..., 0]
    denom = torch.matmul(
        plane_norm[None, :, None, :], rd[:, None, :, None])[..., 0]
    t = t / (denom + eps)
    intersection_global = ro[:, None, :] + t * rd[:, None, :]  # NK3
    plane_origins = -plane_offset * plane_norm
    plane_x = planes[None, :, 3: 6, None]
    plane_y = planes[None, :, 6: 9, None]
    plane_xy = torch.cat([plane_x, plane_y], -1)  # 1K32
    intersection_plane = (intersection_global -
                          plane_origins[None, ...])[..., None, :]  # NK13
    intersections = torch.matmul(
        intersection_plane, plane_xy).squeeze(2)  # NK2
    return intersections, intersection_global, t


def uv_intersection_to_gobal(intersection, planes):
    """convert uv coord to global coord

    Args:
        intersection (FloatTensor): uv coord, [NxKx2], K being number of planes.
        planes (list of FloatTensor): Plane params [K, 10], where a = planes[:,:3], b = planes[:,9] and planes are defined by aTx+b=0. Planes[:,3:6] being plane x directions, Planes[:,6:9] being y directions.
    """
    plane_norm = planes[:, :3]  # Kx3
    plane_offset = planes[:, 9:]  # kx1
    plane_origins = -plane_offset * plane_norm  # kx3
    plane_x = planes[None, :, 3: 6, None]
    plane_y = planes[None, :, 6: 9, None]
    plane_xy = torch.cat([plane_x, plane_y], -1).permute(0, 1, 3, 2)  # 1K23
    # intersection_plane = (intersection_global - plane_origins[None, ...])[..., None, :]  # NK13
    intersections_global = torch.matmul(
        intersection[..., None, :], plane_xy).squeeze(2)  # NK2
    intersections_global += plane_origins[None, ...]
    return intersections_global


def rays_from_poses(pose, K, res, perturb=False):
    """generate rays for intersection test

    Args:
        pose (FloatTensor): camera C2W transformation 4x4
        K (FloatTensor): intrinsics matrix 3x3
        res ([int, int]): resolution, [H,W]

    Returns:
        ro : ray origin, Nx3
        rd : ray direction, Nx3
    """
    H, W = res
    coord = torch.ones([H, W, 3], device=pose.device)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=pose.device), torch.arange(
        W, dtype=torch.float32, device=pose.device))
    if perturb:
        xx = xx + (torch.rand_like(xx) - 0.5)
        yy = yy + (torch.rand_like(yy) - 0.5)
    coord[..., 0] = xx
    coord[..., 1] = yy
    R = pose[: 3, : 3].T
    t = pose[: 3, 3: 4].T
    # depth = depth[..., None]
    ro = t  # 1x3
    rd = torch.matmul(torch.matmul(
        coord, torch.inverse(K).transpose(0, 1)), R)  # HW3
    # rd = torch.renorm(rd, p=2, dim=-1, maxnorm=1)
    rd = rd.reshape([-1, 3])
    ro = ro.expand_as(rd)
    return ro, rd


def unproject_depth_map(pose, K, depth):
    """generate rays for intersection test

    Args:
        pose (FloatTensor): camera C2W transformation 4x4
        K (FloatTensor): intrinsics matrix 3x3
        res ([int, int]): resolution, [H,W]

    Returns:
        ro : ray origin, Nx3
        rd : ray direction, Nx3
    """
    _, _, H, W = depth.shape
    coord = torch.ones([H, W, 3], device=depth.device)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=None, device=depth.device), torch.arange(
        W, dtype=None, device=depth.device))
    coord[..., 0] = xx
    coord[..., 1] = yy
    R = pose[: 3, : 3].T
    t = pose[: 3, 3: 4].T
    # depth = depth[..., None]
    ro = t  # 1x3
    rd = torch.matmul(torch.matmul(
        coord, torch.inverse(K).transpose(0, 1)), R)  # HW3
    # rd = torch.renorm(rd, p=2, dim=-1, maxnorm=1)
    rd = rd.reshape([-1, 3])
    ro = ro.expand_as(rd)
    return ro, rd


def ray_direction_to_plane_incident_vec(rd, plane_params):
    """convert ray direction into incident direction vectors for each plane

    Args:
        rd (FloatTensor): Nx3, not normalized, in world coordinate
        plane_parames (FloatTensor): Kx10, K being number of planes. Note a = planes[:,:3], b = planes[:,9] and planes are defined by aTx+b=0. Planes[:,3:6] being plane x directions, Planes[:,6:9] being y directions.
    return:
        incident_vec (FloatTensor): NxKx3, normalized, in plane coordinate


    """
    plane_coord = plane_params[:, :9].reshape(
        [-1, 3, 3])  # Kx3x3, where [:,0,:] being normal vec, [:,1,:] being x and  [:,2,:] being y.
    rd = rd[:, None, None, :]  # rd as Nx1x1x3
    # normalize rd:
    rd = rd / torch.norm(rd, p=2, dim=3, keepdim=True)
    # import pdb
    # pdb.set_trace()
    # take matmul
    # torch.matmul(plane_coord[None, ...], rd)[..., 0]  # NxKx3
    incident_vec = (rd@plane_coord.permute([0, 2, 1]))[..., 0, :]
    return incident_vec


def cumprod_exclusive_torch(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.
    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.
    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod


def sample_rays_from_poses(pose, K, res, n_samples=32):
    """generate rays for intersection test

    Args:
        pose (FloatTensor): camera C2W transformation 4x4
        K (FloatTensor): intrinsics matrix 3x3
        res ([int, int]): resolution, [H,W]

    Returns:
        ro : ray origin, Nx3
        rd : ray direction, Nx3
    """
    # generate random offsets:

    H, W = res
    #offsets = torch.rand([n_samples, H, W])
    coord = torch.ones([H, W, 3], device=pose.device)
    yy, xx = torch.meshgrid(torch.arange(
        H, dtype=None, device=pose.device), torch.arange(W, dtype=None, device=pose.device))
    coord[..., 0] = xx
    coord[..., 1] = yy
    R = pose[: 3, : 3].T
    t = pose[: 3, 3: 4].T
    # depth = depth[..., None]
    ro = t  # 1x3
    rd_list = []
    for n in range(n_samples):
        offset = torch.rand_like(coord) - 0.5
        offset[..., 2] = 0
        coord_new = coord + offset  # torch.rand([H, W, 3])

        rd = torch.matmul(torch.matmul(
            coord_new, torch.inverse(K).transpose(0, 1)), R)  # HW3
        rd_list.append(rd)
    rd = torch.stack(rd_list, dim=0)  # n_samples x H x W x3
    rd = rd.reshape([-1, 3])
    ro = ro.expand_as(rd)
    return ro, rd


def composite_rays(rgba, depth_samples, eps=1e-6, use_4d_input=False):
    """Composite rays based on depth orderding, with depth culling

    Args:
        rgba (FloatTensor): NxKx(C+1) rgba tensor
        depth_samples (FloatTensor): NxKx1 depth values for each rgba points

    Returns:
        rgb: NxC
        alpha: Nx1
        exp_depth: Nx1

    """
    if use_4d_input:
        H, W, _, _ = rgba.shape
        depth_samples = depth_samples.reshape([H*W, -1, 1])
        rgba = rgba.reshape([H*W, -1, 4])
    depth_sorted, idx = torch.sort(depth_samples, dim=1)
    idx = idx.expand_as(rgba)
    rgba = torch.gather(rgba, 1, idx)
    alpha = rgba[..., -1:]  # NK1
    # depth culling
    alpha_mask = torch.where(depth_sorted > eps, torch.ones_like(
        alpha), torch.zeros_like(alpha))
    alpha = alpha * alpha_mask
    alpha.squeeze_(-1)

    weights = alpha * \
        torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.-alpha], -1), -1)[:, :-1]
    weights.unsqueeze_(-1)
    # stop_prob = torch.roll(torch.cumprod(1.0 - alpha, 1), 1, 1)
    # stop_prob[:, 0, :] = 1.0
    # stop_prob = stop_prob * alpha
    # * alpha
    # stop_prob * cumprod_exclusive_torch(1.0 - alpha + 1e-10)
    # import pdb
    # pdb.set_trace()

    # stop_prob = torch.cumprod(1 - alpha, 1)  # NK1
    #stop_prob = torch.nn.functional.pad(stop_prob[:, : -1, :], (0, 0, 1, 0), mode='constant', value=1.0)
    # stop_prob = alpha * stop_prob
    rgb = torch.sum(rgba[..., : -1] * weights, 1)  # NxC
    alpha = torch.sum(weights, 1)  # Nx1
    exp_depth = torch.sum(depth_sorted * weights, 1)  # Nx1
    return rgb, alpha, exp_depth


def rd_to_spherical_coord(ro, rd, radius=1000):
    """ray direction to spherical coordinate

    Args:
        rd (FloatTensor): ray direction tensor, Nx3

    Returns:
        sph_coord: FloatTensor, Nx2 as theta, phi per ray in global coord
    """
    # solve for \alpha, where ro+rd*\alpha lies on the sphere.

    b = torch.sum(ro*rd, dim=-1)*2
    c = torch.sum(ro**2, dim=-1) - radius**2
    alpha = 0.5*((b**2 - 4*c)**0.5 - b)
    sph_coord = ro+alpha[..., None]*rd
    sph_coord = sph_coord / torch.norm(sph_coord, dim=-1, keepdim=True)
    return sph_coord
