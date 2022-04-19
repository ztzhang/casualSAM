import moderngl
import types
import numpy as np
import torch
from torch import nn
from time import time
from util.util_gl_camera import perspective_from_intrinsics
from glob import glob
from os.path import join


def to_tensor(tensor):
    if type(tensor) == np.ndarray:
        return torch.from_numpy(tensor).float()
    else:
        return tensor.float()


def np_float(data):
    """Returns data as a 32-bit float numpy array."""
    return np.array(data, dtype=np.float32)


def update_gl_context(gl, width, height):
    size = (width, height)
    gl.resolution = size
    gl.fbo.release()
    gl.depth_texture_in.release()
    id_renderbuffer = gl.context.renderbuffer(size, components=1, dtype='i4')
    bary_renderbuffer = gl.context.renderbuffer(size, components=3, dtype='f4')
    depth_renderbuffer = gl.context.depth_renderbuffer(size)
    depth_texture_out = gl.context.texture(size, components=1, dtype='f4')
    depth_texture_in = gl.context.texture(size, components=1, dtype='f4')
    gl.fbo = gl.context.framebuffer(
        (id_renderbuffer, bary_renderbuffer, depth_texture_out), depth_renderbuffer)
    gl.fbo.use()
    gl.depth_texture_in = depth_texture_in

    # Create the shaders that render triangle ids and vertex barycentric
    # coordinates for each pixel.
    gl.shader['screenSize'].value = size
    gl.initialized = True
    # Make the Vertex Array Object for the triangles of the mesh.
    return gl
# this changed the gl resolution


def create_gl_context(width, height):
    """Returns OpenGL context for rasterizing ids and barycentric coordinates."""
    gl = types.SimpleNamespace()
    gl.width = width
    gl.height = height

    # Create the context.
    gl.context = moderngl.create_context(standalone=True, backend='egl')
    gl.context.enable(moderngl.DEPTH_TEST)

    # Create the renderbuffers.

    size = (width, height)
    gl.resolution = size
    id_renderbuffer = gl.context.renderbuffer(size, components=1, dtype='i4')
    bary_renderbuffer = gl.context.renderbuffer(size, components=3, dtype='f4')
    depth_renderbuffer = gl.context.depth_renderbuffer(size)
    depth_texture_out = gl.context.texture(size, components=1, dtype='f4')
    depth_texture_in = gl.context.texture(size, components=1, dtype='f4')
    gl.fbo = gl.context.framebuffer(
        (id_renderbuffer, bary_renderbuffer, depth_texture_out), depth_renderbuffer)
    gl.fbo.use()
    gl.depth_texture_in = depth_texture_in

    # Create the shaders that render triangle ids and vertex barycentric
    # coordinates for each pixel.
    gl.shader = gl.context.program(
        vertex_shader="""
            # version 330
            uniform mat4 camera_matrix;
            in vec4 in_vertex;
            in vec3 in_barys;
            out vec3 barys;
            void main() {
                barys = in_barys;
                gl_Position = camera_matrix * in_vertex;
            }
  """,
        fragment_shader="""
            # version 330
            uniform sampler2D depthTexture;
            uniform vec2 screenSize;
            in vec3 barys;
            out int out_id;
            out vec3 out_barys;
            out float outDepth;
            void main() {
                float fragDepth = gl_FragCoord.z;
                float minDepth = texture(depthTexture, gl_FragCoord.xy / screenSize).r;
                if (fragDepth < minDepth + 1e-4) { discard; }
                out_id = gl_PrimitiveID + 1;
                out_barys = barys;
                outDepth = fragDepth;

            }
  """)
    gl.shader['screenSize'].value = size
    gl.initialized = True
    # Make the Vertex Array Object for the triangles of the mesh.
    return gl


def create_gl_buffer(gl, triangles):
    if torch.is_tensor(triangles):
        triangles = triangles.detach().cpu().numpy()
    arr_in = triangles.reshape((-1)).astype('f4')
    arr_buf = gl.context.buffer(arr_in)
    vao = gl.context.simple_vertex_array(
        gl.shader, arr_buf, 'in_vertex', 'in_barys')
    return vao, arr_buf


def render_per_batch(gl, vao, world2clip, width, height, num_layers=1, zeros=None):
    if gl.height == height and gl.width == width and gl.initialized:
        pass
    else:
        gl = create_gl_context(width, height)
    gl.timing = time()

    if zeros is None:
        zeros = np.zeros(height * width, dtype=np.float32)
    layered_images_tri_id = []
    layered_images_bary = []
    gl.shader['camera_matrix'] = tuple(world2clip.transpose().flatten())
    # layered_images_depth = []

    # arr_in = triangles[i, ...].reshape((-1)).astype('f4')
    gl.fbo.clear(depth=1.0)
    gl.context.enable(moderngl.DEPTH_TEST)
    # gl_context.ctx.enable(moderngl.CULL_FACE)
    # arr_buf = gl.context.buffer(arr_in)
    # vao = gl.context.simple_vertex_array(gl.shader, arr_buf, 'in_vertex', 'in_barys')

    # On the first peel, previous depth is 0.
    gl.depth_texture_in.write(zeros.data)
    gl.depth_texture_in.use()
    accumulated_depth = np.zeros((height, width), dtype=np.float32)

    tri_layers = []
    bary_layers = []

    for layer_idx in range(num_layers):
        vao.render(mode=moderngl.TRIANGLES)
        tri_buffer = gl.fbo.read(components=1, attachment=0, dtype='i4')
        tri_arr = np.frombuffer(tri_buffer, dtype=np.int32)
        tri_layers.append(tri_arr.reshape((height, width)))

        bary_buf = gl.fbo.read(components=3, attachment=1, dtype='f4')
        bary_arr = np.frombuffer(
            bary_buf, dtype=np.float32).reshape(height, width, 3)
        bary_layers.append(bary_arr)

        depth_buffer = gl.fbo.read(components=1, attachment=2, dtype='f4')
        gl.fbo.clear(depth=1.0)
        # Write the last layer's depth into the input
        depth_this_layer = np.frombuffer(
            depth_buffer, dtype=np.float32).reshape((height, width))

        accumulated_depth = np.maximum(depth_this_layer, accumulated_depth)

        gl.depth_texture_in.write(accumulated_depth.data)  # depth_buffer)

    layered_images_tri_id = np.stack(tri_layers, axis=-1)

    layered_images_bary = np.stack(bary_layers, axis=-1)

    tri_mask = layered_images_tri_id.reshape([height, width, num_layers])
    tri_mask = np.where(tri_mask == 0, 0., 1.)
    gl.timing = time() - gl.timing
    return layered_images_tri_id, layered_images_bary, tri_mask


def render(gl, triangles, world2clip, width, height, num_layers=1):
    if gl.height == height and gl.width == width and gl.initialized:
        pass
    else:
        gl = create_gl_context(width, height)
    gl.timing = time()
    if torch.is_tensor(triangles):
        triangles = triangles.detach().cpu().numpy()
    bs = triangles.shape[0]
    assert len(triangles.shape) == 3
    zeros = np.zeros(height * width, dtype=np.float32)
    layered_images_tri_id = []
    layered_images_bary = []
    gl.shader['camera_matrix'] = tuple(
        world2clip.cpu().numpy().transpose().flatten())
    # layered_images_depth = []
    for i in range(bs):
        arr_in = triangles[i, ...].reshape((-1)).astype('f4')
        gl.fbo.clear(depth=1.0)
        gl.context.enable(moderngl.DEPTH_TEST)
        # gl_context.ctx.enable(moderngl.CULL_FACE)
        arr_buf = gl.context.buffer(arr_in)
        vao = gl.context.simple_vertex_array(
            gl.shader, arr_buf, 'in_vertex', 'in_barys')

        # On the first peel, previous depth is 0.
        gl.depth_texture_in.write(zeros.data)
        gl.depth_texture_in.use()
        accumulated_depth = np.zeros((height, width), dtype=np.float32)

        tri_layers = []
        bary_layers = []

        for layer_idx in range(num_layers):
            vao.render(mode=moderngl.TRIANGLES)
            tri_buffer = gl.fbo.read(components=1, attachment=0, dtype='i4')
            tri_arr = np.frombuffer(tri_buffer, dtype=np.int32)
            tri_layers.append(tri_arr.reshape((height, width)))

            bary_buf = gl.fbo.read(components=3, attachment=1, dtype='f4')
            bary_arr = np.frombuffer(
                bary_buf, dtype=np.float32).reshape(height, width, 3)
            bary_layers.append(bary_arr)

            depth_buffer = gl.fbo.read(components=1, attachment=2, dtype='f4')
            gl.fbo.clear(depth=1.0)
            # Write the last layer's depth into the input
            depth_this_layer = np.frombuffer(
                depth_buffer, dtype=np.float32).reshape((height, width))

            accumulated_depth = np.maximum(depth_this_layer, accumulated_depth)

            gl.depth_texture_in.write(accumulated_depth.data)  # depth_buffer)

        layered_images_tri_id.append(np.stack(tri_layers, axis=-1))

        layered_images_bary.append(np.stack(bary_layers, axis=-1))

        arr_buf.release()
        vao.release()
        del arr_buf
        del vao
        # del buf
    # arrs = np.stack(arrs)
    tri_id_batch = np.stack(layered_images_tri_id)
    bary_batch = np.stack(layered_images_bary)
    # depth_batch = np.stack(layered_images_depth)
    tri_mask = tri_id_batch.reshape([bs, height, width, num_layers])
    tri_mask = np.where(tri_mask == 0, 0., 1.)
    gl.timing = time() - gl.timing
    return tri_id_batch, bary_batch, tri_mask

# %%
# here I need a util function to convert camera matrices into opengl format.


def depth_to_points(depth, cam_c2w, K):
    """unproject depth as layers

    Args:
        depth (torch.floatTensor): depth map
        cam_c2w (torch.FloatTensor): camera to world matrices
        K (torch.FloatTensor): camera intrinsics

    Returns:
        pt_global: global xyzs for each pixel
    """
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
    return pt_global  # .reshape([-1, 3])


def create_triangles(h, w):
    """Creates mesh triangle indices from a given pixel grid size.

       This function is not and need not be differentiable as triangle indices are
       fixed.

    Args:
      h: (int) denoting the height of the image.
      w: (int) denoting the width of the image.

    Returns:
      triangles: 2D numpy array of indices (int) with shape (2(W-1)(H-1) x 3)
    """
    x, y = np.meshgrid(range(w - 1), range(h - 1))
    tl = y * w + x
    tr = y * w + x + 1
    bl = (y + 1) * w + x
    br = (y + 1) * w + x + 1
    triangles = np.array([tl, bl, tr, br, tr, bl])
    triangles = np.transpose(triangles, (1, 2, 0)).reshape(
        ((w - 1) * (h - 1) * 2, 3))
    return triangles


def mask_out_triangles(triangles, mask):
    assert len(mask.shape) == 1
    valid = mask[triangles]
    valid = np.sum(valid, -1)  # K
    valid_idx = np.where(valid == 0)[0]
    filtered_triangles = triangles[valid_idx, :]
    return filtered_triangles


def convert_to_gl_intrinsics(intrinsics, debug=False):
    """Computes a perspective matrix from camera intrisics.

    Check this for the convention used for intrinsics:
    https://cs.corp.google.com/piper///depot/google3/research/vision/parallax/learning/lib/geometry.py?type=cs&q=f:parallax.*geometry.py+intrinsics&g=0&l=26

    Args:
      intrinsics: [B, 4] Source camera intrinsics tensor (f_x, f_y, c_x, c_y)

    Returns:
      A [B, 4, 4] float32 Tensor that maps from right-handed points in
      camera space to left-handed points in clip space.
    """
    focal_x = intrinsics[:, 0]
    focal_y = intrinsics[:, 1]
    # Parallax has 0.5, 0.5 at center, OpenGL has 0 at center.
    center_x = intrinsics[:, 2] - 0.5
    center_y = intrinsics[:, 3] - 0.5
    near_clip = 0.0001 * torch.ones_like(focal_x)
    far_clip = 10000.0 * torch.ones_like(focal_x)

    proj = perspective_from_intrinsics(focal_x, focal_y, center_x,
                                       center_y, near_clip, far_clip,
                                       1, 1)
    return proj


# this function needs to be updated for handeling tensors
def construct_world2clip(K, camera2world, H, W):
    intrinsics = np.asarray((K[0, 0] / W,
                             K[1, 1] / H,
                             K[0, 2] / (W - 1),
                             K[1, 2] / (H - 1)))

    intrinsics = torch.from_numpy(intrinsics.astype(np.float32))
    cam2clip = convert_to_gl_intrinsics(intrinsics[None, ...])
    world2cam = torch.from_numpy(
        np.linalg.inv(camera2world).astype(np.float32))
    cam2gl = torch.eye(4)
    cam2gl[1, 1] *= 1.
    cam2gl[2, 2] *= -1.

    world2gl = torch.matmul(cam2gl, world2cam)
    world2clip = torch.matmul(cam2clip, world2gl)
    return world2clip


def interpolate_no_frag(attr_per_vertex, frag_id, mask, barycentrics, cam_w2c_4x4_T, triangles_pos):
    # attr_per_vertex = vertex_attr[H_ids, W_ids, :]
    corner_attr = attr_per_vertex[frag_id, ...]  # H, W, L, Tri, Color
    corner_attr = corner_attr.permute([0, 1, 3, 4, 2])  # H, W, Tri, color, L
    rendered_vertex_attr = torch.sum(
        barycentrics[..., None, :] * corner_attr, 2)  # * mask

    # # we also need depth here
    global_pos = triangles_pos[frag_id, :].permute(
        [0, 1, 3, 4, 2])  # H,W,tri,pos,L
    global_pos = torch.sum(
        barycentrics[..., None, :] * global_pos, 2).permute([0, 1, 3, 2])  # H, W, L, Pos
    cam_pos = torch.matmul(global_pos, cam_w2c_4x4_T)  # x,y,z,w
    depth = (cam_pos[..., 2:3] / (cam_pos[..., 3:] + 1e-6)
             ).permute([0, 1, 3, 2])
    depth = depth * (mask) + (mask - 1)  # negetive depth for invalid regions.

    return rendered_vertex_attr, depth, mask


def interpolate(attr_per_vertex, frag_id, mask, barycentrics, cam_w2c_4x4_T, triangles_pos, frag_attr):
    # attr_per_vertex = vertex_attr[H_ids, W_ids, :]
    corner_attr = attr_per_vertex[frag_id, ...]  # H, W, L, Tri, Color
    corner_attr = corner_attr.permute([0, 1, 3, 4, 2])  # H, W, Tri, color, L
    rendered_vertex_attr = torch.sum(
        barycentrics[..., None, :] * corner_attr, 2)  # * mask
    rendered_frag_attr = frag_attr[frag_id, :].permute(
        [0, 1, 3, 2]) * mask  # H,W,C,L
    # # we also need depth here
    global_pos = triangles_pos[frag_id, :].permute(
        [0, 1, 3, 4, 2])  # H,W,tri,pos,L
    global_pos = torch.sum(
        barycentrics[..., None, :] * global_pos, 2).permute([0, 1, 3, 2])  # H, W, L, Pos
    cam_pos = torch.matmul(global_pos, cam_w2c_4x4_T)  # x,y,z,w
    depth = (cam_pos[..., 2:3] / (cam_pos[..., 3:] + 1e-6)
             ).permute([0, 1, 3, 2])
    depth = depth * (mask) + (mask - 1)  # negetive depth for invalid regions.

    return rendered_vertex_attr, rendered_frag_attr, depth, mask


class mesh_with_attributes(nn.Module):
    def __init__(self, gl, img_size, depth, K, cam_c2w, mask, vertex_attr_ch=3, fragment_attr_ch=1, num_layers=2, use_latent_texture=False, no_frag_attr=False):
        super().__init__()
        self.mesh_geometry = mesh_with_gl_context(
            gl, img_size, depth, K, cam_c2w, mask)
        H, W = img_size
        if not use_latent_texture:
            self.texture_buffer = nn.Parameter(
                torch.zeros([1, vertex_attr_ch, H, W]))
        else:
            self.texture_buffer = None

        # this should use depth resolution.
        depth_h, depth_w = depth.shape

        ww = torch.linspace(-1, 1, depth_w)
        hh = torch.linspace(-1, 1, depth_h)
        yy, xx = torch.meshgrid(hh, ww)
        triangle_vertex_uv_coord = torch.stack([xx, yy], -1)  # H, W ,2
        H_ids = self.mesh_geometry.image_index_H
        W_ids = self.mesh_geometry.image_index_W
        uv_per_vertex = triangle_vertex_uv_coord[H_ids, W_ids, :]

        self.register_buffer('uv_per_vertex', uv_per_vertex)
        self.no_frag = no_frag_attr
        if not self.no_frag:
            self.frag_attr = nn.Parameter(torch.zeros(
                self.mesh_geometry.triangles_pos.shape[0], fragment_attr_ch))
        # self.register_buffer('pt_global', self.mesh_geometry.triangles_pos)  # Nx3
        self.register_buffer('cam_w2c_4x4_T', torch.from_numpy(
            np.linalg.inv(cam_c2w).T).float())
        self.jitted_interpolation = None

        self.num_layers = num_layers
        self.use_latent_texture = use_latent_texture

    def move_to_gl(self):
        self.mesh_geometry.create_gl_buffer()

    def free_from_gl(self):
        self.mesh_geometry.release_gl_buffer()

    def init_frag_attr(self, frag_attr):
        self.frag_attr.data = to_tensor(frag_attr)

    def init_vertex_attr(self, vertex_attr, init_alpha):
        vertex_attr = to_tensor(vertex_attr).permute(2, 0, 1)[None, ...]
        if vertex_attr.shape[1] == 4:
            self.texture_buffer.data = vertex_attr
        else:
            self.texture_buffer.data[:, :3, ...] = vertex_attr
            self.texture_buffer.data[:, 3, ...] = init_alpha

    def init_vertex_attr_raw(self, vertex_attr):
        vertex_attr = to_tensor(vertex_attr).permute(2, 0, 1)[None, ...]
        self.texture_buffer.data = vertex_attr

    def init_frag_attr_scalar(self, frag_attr_scalar):
        self.frag_attr.data.fill_(frag_attr_scalar)

    def forward(self, world2clip, input_texture=None):
        assert(self.use_latent_texture ^ (input_texture is None))
        frag_id, barycentrics, mask = self.mesh_geometry.rasterize(
            world2clip, self.num_layers)
        barycentrics = torch.from_numpy(barycentrics).float().to(
            device=self.cam_w2c_4x4_T.device, non_blocking=True)
        mask = torch.from_numpy(mask).float().to(
            device=self.cam_w2c_4x4_T.device, non_blocking=True)[..., None, :]
        frag_id = torch.from_numpy(frag_id).long().to(
            device=self.cam_w2c_4x4_T.device, non_blocking=True)
        frag_id -= 1

        if self.jitted_interpolation is None:
            if not self.no_frag:
                self.jitted_interpolation = torch.jit.trace(
                    interpolate,
                    (self.uv_per_vertex, frag_id, mask, barycentrics,
                     self.cam_w2c_4x4_T, self.mesh_geometry.triangles_pos, self.frag_attr))
            else:
                self.jitted_interpolation = torch.jit.trace(
                    interpolate_no_frag,
                    (self.uv_per_vertex, frag_id, mask,
                     barycentrics, self.cam_w2c_4x4_T, self.mesh_geometry.triangles_pos))

        if self.no_frag:
            rendered_vertex_attr_uv, depth, mask = self.jitted_interpolation(
                self.uv_per_vertex, frag_id, mask, barycentrics, self.cam_w2c_4x4_T, self.mesh_geometry.triangles_pos)
            rendered_frag_attr = None
        else:
            rendered_vertex_attr_uv, rendered_frag_attr, depth, mask = self.jitted_interpolation(
                self.uv_per_vertex, frag_id, mask, barycentrics, self.cam_w2c_4x4_T, self.mesh_geometry.triangles_pos, self.frag_attr)
        sample_grid_location = rendered_vertex_attr_uv.permute([3, 0, 1, 2])

        if input_texture is not None:
            texture_buffer = input_texture
        else:
            texture_buffer = self.texture_buffer
        sampled_texture = torch.nn.functional.grid_sample(
            texture_buffer.expand([self.num_layers, -1, -1, -1]),
            sample_grid_location,
            align_corners=True).permute([2, 3, 1, 0])  # L,C,H,W -> H,W,C,L
        sampled_texture = sampled_texture*mask
        return sampled_texture, rendered_frag_attr, depth, mask

# let's jit this


class mesh_with_gl_context(nn.Module):
    def __init__(self, gl, img_size, depth, K, cam_c2w, mask):
        super().__init__()
        self.gl = gl
        self.img_size = img_size
        H, W = img_size

        depth_h, depth_w = depth.shape
        triangles = create_triangles(depth_h, depth_w)
        self.triangles = triangles
        # this should convert K into depth resolution
        depth_k = K.copy()
        depth_k[0, :] /= W / depth_w
        depth_k[1, :] /= H / depth_h

        # this has to match depth resolution
        pt_global = depth_to_points(depth, cam_c2w, depth_k)
        filtered_triangles = mask_out_triangles(triangles, mask.reshape([-1]))
        pt_global_flat = pt_global.reshape([-1, 3])
        pt_global_flat = np.pad(
            pt_global_flat, ((0, 0), (0, 1)), constant_values=1)
        triangles_pos = pt_global_flat[filtered_triangles, :]
        self.register_buffer(
            'triangles_pos', torch.from_numpy(triangles_pos).float())
        triangles_pos = triangles_pos.reshape([-1, 4])
        barys = np.tile(np.eye(3, dtype=np.float32),
                        (triangles_pos.shape[0] // 3, 1))
        image_index_H, image_index_W = np.unravel_index(
            filtered_triangles.reshape([-1, 3]), (depth_h, depth_w))
        self.register_buffer(
            'image_index_H', torch.from_numpy(image_index_H).long())
        self.register_buffer(
            'image_index_W', torch.from_numpy(image_index_W).long())

        triangles_pos = torch.from_numpy(
            triangles_pos.astype(np.float32))[None, ...]
        barys = torch.from_numpy(barys[None, ...]).float()
        self.triangles_pos_aug = torch.cat((triangles_pos, barys), -1)
        self.vao = None
        self.arr_buf = None
        self.gl_buffer_ready = False
        self.zeros = np.zeros(H * W, dtype=np.float32)

    def update_geometry(self, depth, K, cam_c2w, mask):
        3
        if self.gl_buffer_ready:
            self.release_gl_buffer()
        H, W = self.img_size
        depth_k = K.copy()
        depth_h, depth_w = depth.shape

        depth_k[0, :] /= W / depth_w
        depth_k[1, :] /= H / depth_h
        triangles = self.triangles
        # this has to match depth resolution
        pt_global = depth_to_points(depth, cam_c2w, depth_k)
        filtered_triangles = mask_out_triangles(triangles, mask.reshape([-1]))
        pt_global_flat = pt_global.reshape([-1, 3])
        pt_global_flat = np.pad(
            pt_global_flat, ((0, 0), (0, 1)), constant_values=1)
        triangles_pos = pt_global_flat[filtered_triangles, :]
        self.triangles_pos.data = torch.from_numpy(triangles_pos).float()

        triangles_pos = triangles_pos.reshape([-1, 4])
        barys = np.tile(np.eye(3, dtype=np.float32),
                        (triangles_pos.shape[0] // 3, 1))
        triangles_pos = torch.from_numpy(
            triangles_pos.astype(np.float32))[None, ...]
        barys = torch.from_numpy(barys[None, ...]).float()
        self.triangles_pos_aug = torch.cat((triangles_pos, barys), -1)
        self.create_gl_buffer()

    def create_gl_buffer(self):
        self.vao, self.arr_buf = create_gl_buffer(
            self.gl, self.triangles_pos_aug[0, ...])
        self.gl_buffer_ready = True

    def release_gl_buffer(self):
        self.vao.release()
        self.arr_buf.release()
        del self.vao
        del self.arr_buf
        self.vao = None
        self.arr_buf = None
        self.gl_buffer_ready = False

    def rasterize(self, world2clip, num_layers=3):
        assert self.gl_buffer_ready
        # self.create_gl_buffer()
        return render_per_batch(self.gl, self.vao, world2clip, self.img_size[1], self.img_size[0], num_layers, zeros=self.zeros)
        # get camera space depth:
        # should this move to gpu?


class MeshCollection(nn.Module):
    # this method will cache the vao for gl context, creating new ones and releasing old ones if queue limit is reached.
    def __init__(self, H, W, max_queue_size=10000, n_layer=2, use_latent_texture=False, no_frag_attr=False):
        # create gl context
        super().__init__()
        self.gl = create_gl_context(W, H)  # this is for rendering.
        self.image_size = [H, W]
        self.img_size = [H, W]
        self.active_queue_id = []
        self.meshes = []
        self.fg_meshes = []
        self.n_layer = n_layer
        self.mesh_cnt = 0
        self.fg_mesh_cnt = 0
        self.max_queue_size = max_queue_size
        self.use_latent_texture = use_latent_texture
        self.no_frag_attr = no_frag_attr

    def initialize_from_single_path(self, frame_path, n_keyframes=10, init_alpha=None):

        if init_alpha is None:
            init_alpha = 1 / n_keyframes

        frame_npzs = sorted(glob(join(frame_path, '*.npz')))
        kf_idx = np.linspace(0, len(frame_npzs) - 1, n_keyframes, dtype=int)

        for idx in kf_idx:
            d = np.load(frame_npzs[idx], allow_pickle=True)
            mask = d['motion_seg']
            depth = d['depth_mvs'].squeeze()
            invalid_depth = np.where(depth < 1e-3, True, False)
            import scipy.ndimage
            invalid_depth = scipy.ndimage.binary_dilation(
                invalid_depth, iterations=20).astype(mask.dtype)
            mask = np.clip(invalid_depth+mask, a_max=1, a_min=0)
            self.register_mesh(depth.squeeze(
            ), d['intrinsics'], d['pose_c2w'], mask, d['img'], init_alpha)

    def initialize_from_path(self, frame_path, depth_path=None, n_keyframes=10, init_alpha=None):
        if depth_path is None:
            depth_path = frame_path

        if init_alpha is None:
            init_alpha = 1 / n_keyframes

        infer_results = sorted(glob(join(depth_path, 'batch*.npz')))

        frame_npzs = sorted(glob(join(frame_path, '*.npz')))
        kf_idx = np.linspace(0, len(infer_results) - 1, n_keyframes, dtype=int)

        for idx in kf_idx:
            d = np.load(frame_npzs[idx], allow_pickle=True)
            f = np.load(infer_results[idx], allow_pickle=True)
            self.register_mesh(f['depth'].squeeze(
            ), d['intrinsics'], d['pose_c2w'], d['motion_seg'], d['img'], init_alpha)

    def register_mesh(self, depth, K, cam_c2w, mask, rgb=None, init_alpha=1.0):
        module = mesh_with_attributes(
            self.gl, self.img_size, depth, K, cam_c2w, mask, vertex_attr_ch=4, num_layers=self.n_layer, use_latent_texture=self.use_latent_texture, no_frag_attr=self.no_frag_attr)
        if rgb is not None and not self.use_latent_texture:
            module.init_vertex_attr(rgb, init_alpha)
        if not self.no_frag_attr:
            module.init_frag_attr_scalar(init_alpha)
        self.add_module(f'mesh_{self.mesh_cnt:04d}', module)
        self.meshes.append(module)
        if len(self.active_queue_id) < self.max_queue_size:
            self.active_queue_id.append(self.mesh_cnt)
            module.move_to_gl()
        self.mesh_cnt += 1

    def register_fg_rgba_mesh(self, depth, K, cam_c2w, mask, rgba):
        module = mesh_with_attributes(
            self.gl, self.img_size, depth, K, cam_c2w, mask, vertex_attr_ch=4, num_layers=self.n_layer, use_latent_texture=self.use_latent_texture, no_frag_attr=self.no_frag_attr)
        if rgba is not None and not self.use_latent_texture:
            module.init_vertex_attr_raw(rgba)
        if not self.no_frag_attr:
            module.init_frag_attr_scalar(0)
        self.add_module(f'mesh_fg_{self.fg_mesh_cnt}', module)
        self.fg_meshes.append(module)
        # if len(self.active_queue_id) < self.max_queue_size:
        # self.active_queue_id.append(self.mesh_cnt)
        module.move_to_gl()
        self.fg_mesh_cnt += 1

    # def register_rgba_mesh(self, depth, K, cam_c2w, mask, rgba):
    #     module = mesh_with_attributes(
    #         self.gl, self.img_size, depth, K, cam_c2w, mask, vertex_attr_ch=4, num_layers=self.n_layer, use_latent_texture=self.use_latent_texture, no_frag_attr=self.no_frag_attr)
    #     if rgba is not None and not self.use_latent_texture:
    #         module.init_vertex_attr_raw(rgba)
    #     if not self.no_frag_attr:
    #         module.init_frag_attr_scalar(0)
    #     self.add_module(f'mesh_{self.mesh_cnt:04d}', module)
    #     self.meshes.append(module)
    #     if len(self.active_queue_id) < self.max_queue_size:
    #         self.active_queue_id.append(self.mesh_cnt)
    #         module.move_to_gl()
    #     self.mesh_cnt += 1

    def render_fg_layer(self, K, cam_c2w, index=0, norm_fn=None, input_texture_maps=None):

        H, W = self.image_size
        assert (K.shape) == (3, 3)
        assert (cam_c2w.shape) == (4, 4)

        world2clip = construct_world2clip(
            K=K, camera2world=cam_c2w, H=H, W=W).cpu().numpy()
        # this is buggy rightnow, where the queue is ignored, assuming active queue contains all stuff

        m = self.fg_meshes[index]
        if input_texture_maps is not None:
            rendered_vertex_attr, rendered_frag_attr, depth, mask = m.forward(
                world2clip, input_texture_maps)
        else:
            rendered_vertex_attr, rendered_frag_attr, depth, mask = m.forward(
                world2clip, None)

        rendered_vertex_attr *= mask

        if self.no_frag_attr:
            rendered_frag_attr = None

        return rendered_vertex_attr[None, ...], rendered_frag_attr, depth[None, ...], mask[None, ...]

    def render_all_mesh_layers(self, K, cam_c2w, norm_fn=None, input_texture_maps=None):
        if self.use_latent_texture:
            assert(input_texture_maps is not None)
            assert(input_texture_maps.shape[0] == len(self.meshes))
        else:
            assert(input_texture_maps is None)
            # input_texture_maps = [None] * len(self.meshes)

        H, W = self.image_size
        assert (K.shape) == (3, 3)
        assert (cam_c2w.shape) == (4, 4)

        world2clip = construct_world2clip(
            K=K, camera2world=cam_c2w, H=H, W=W).cpu().numpy()
        # this is buggy rightnow, where the queue is ignored, assuming active queue contains all stuff
        rendered_vertex_attr_list = []
        rendered_frag_attr_list = []
        depth_list = []
        mask_list = []
        for idm, m in enumerate(self.meshes):
            if input_texture_maps is not None:
                rendered_vertex_attr, rendered_frag_attr, depth, mask = m.forward(
                    world2clip, input_texture_maps[idm:idm+1, ...])
            else:
                rendered_vertex_attr, rendered_frag_attr, depth, mask = m.forward(
                    world2clip, None)
            rendered_vertex_attr_list.append(rendered_vertex_attr)
            rendered_frag_attr_list.append(rendered_frag_attr)
            depth_list.append(depth)
            mask_list.append(mask)
        rendered_vertex_attr_list = torch.stack(rendered_vertex_attr_list, 0)
        if self.no_frag_attr:
            rendered_frag_attr_list = None
        else:
            rendered_frag_attr_list = torch.stack(rendered_frag_attr_list, 0)
        depth_list = torch.stack(depth_list, 0)
        mask_list = torch.stack(mask_list, 0)
        if norm_fn is not None:
            rendered_vertex_attr_list = norm_fn(
                rendered_vertex_attr_list)*mask_list
            if not self.no_frag_attr:
                rendered_frag_attr_list = norm_fn(
                    rendered_frag_attr_list)*mask_list
        return rendered_vertex_attr_list, rendered_frag_attr_list, depth_list, mask_list
# %%
