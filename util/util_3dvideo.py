import numpy as np
from .util_3dvideo_template import video3d_template


def save_ply(result_path, name, ptcld, colors, mask=None):
    # colors = (ptcld - np.min(ptcld, axis=0)) / (np.max(ptcld, axis=0) - np.min(ptcld, axis=0))
    colors = np.floor(colors * 255).astype(np.uint8)
    template = "ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    template = template % (ptcld.shape[0])
    if mask is not None:
        for p, color, v in zip(ptcld, colors, mask):
            if v:
                template += "%f %f %f %d %d %d \n" % (
                    p[0], p[1], p[2], color[0], color[1], color[2])
    else:
        for p, color in zip(ptcld, colors):
            template += "%f %f %f %d %d %d \n" % (
                p[0], p[1], p[2], color[0], color[1], color[2])
    with open(f'{result_path}/{name}.ply', 'w') as f:
        f.write(template)


def save_line_ply(result_path, name, points, edges):
    # colors = (ptcld - np.min(ptcld, axis=0)) / (np.max(ptcld, axis=0) - np.min(ptcld, axis=0))

    template = "ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n element edge %d \n property int32 vertex1\n property int32 vertex2\nend_header\n"
    template = template % (points.shape[0], edges.shape[0])
    for p in points:
        template += "%f %f %f\n" % (p[0], p[1], p[2])
    for e in edges:
        template += "%d %d\n" % (e[0], e[1])
    with open(f'{result_path}/{name}.ply', 'w') as f:
        f.write(template)


def depth_to_points(depth, cam_c2w, K):
    H, W = depth.shape
    coord = np.ones([H, W, 3])
    xx, yy = np.meshgrid(np.arange(W).astype(float),
                         np.arange(H).astype(float))
    coord[..., 0] = xx
    coord[..., 1] = yy
    R = cam_c2w[:3, :3].T
    t = cam_c2w[:3, 3:4].T
    depth = depth[..., None]
    pt_global = (depth * (coord @ np.linalg.inv(K).T)) @ R + t
    return pt_global.reshape([-1, 3])


def depth_to_points_Rt(depth, c2w_R, c2w_t, K):
    H, W = depth.shape
    coord = np.ones([H, W, 3])
    xx, yy = np.meshgrid(np.arange(W).astype(float),
                         np.arange(H).astype(float))
    coord[..., 0] = xx
    coord[..., 1] = yy
    R = c2w_R.T
    t = c2w_t.T
    depth = depth[..., None]
    pt_global = (depth * (coord @ np.linalg.inv(K).T)) @ R + t
    return pt_global.reshape([-1, 3])


class Video_3D_Webpage:
    def __init__(self):
        self.web_template = video3d_template
        self.ply_paths = []

    def add_ply_by_path(self, ply_path):
        self.ply_paths.append(ply_path)

    def save_to_file(self, path):
        ply_string = ''
        for x in self.ply_paths:
            ply_string += '"' + x + '",'
        content = self.web_template.format(ply_string=ply_string)
        with open(path, 'w') as f:
            f.write(content)

# project points to camera space.
