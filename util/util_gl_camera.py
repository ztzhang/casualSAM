import torch


def ortho(left, right, bottom, top, near_val, far_val):
    """Computes an orthographic camera transformation matrix.

    Functionality mimes glOrtho (gl/include/GL/gl.h).

    Args:
      left: Tensor or value convertible to Tensor, with shape [batch_size]
        specifying location of left clipping plane.
      right: Tensor (or convertible value) with shape [batch_size] specifying
        location of right clipping plane.
      bottom: Tensor (or convertible value) with shape [batch_size] specifying
        location of bottom clipping plane.
      top: Tensor (or convertible value) with shape [batch_size] specifying
        location of top clipping plane.
      near_val: Tensor (or convertible value) with shape [batch_size] specifying
        location of near clipping plane.
      far_val: Tensor (or convertible value) with shape [batch_size] specifying
        location of far clipping plane.

    Returns:
       A [batch_size, 4, 4] float tensor that maps from right-handed points in
         eye space to left-handed points in clip space.
    """
    left = to_tensor(left)
    right = to_tensor(right)
    bottom = to_tensor(bottom)
    top = to_tensor(top)
    near_val = to_tensor(near_val)
    far_val = to_tensor(far_val)

    z = torch.zeros_like(left)
    o = torch.ones_like(left)
    depth_range = far_val - near_val
    # pyformat: disable
    m = torch.stack([
        2.0 / (right - left), z, z, -(right + left) / (right - left),
        z, 2.0 / (top - bottom), z, -(top + bottom) / (top - bottom),
        z, z, -2.0 / (depth_range), -(far_val + near_val) / (depth_range),
        z, z, z, o
    ],
        dim=1)
    # pyformat: enable
    return torch.reshape(m, [-1, 4, 4])


def to_tensor(maybe_t, dtype=None):
    if torch.is_tensor(maybe_t):
        if dtype is None or maybe_t.dtype == dtype:
            return maybe_t
        return maybe_t.type(dtype)
    if dtype is None:
        return torch.tensor(maybe_t).cuda()
    return torch.tensor(maybe_t, dtype=dtype).cuda()


def perspective_from_intrinsics(focal_x, focal_y, center_offset_x,
                                center_offset_y, near_clip, far_clip,
                                image_width, image_height, debug=False):
    """Computes a perspective matrix from vision-style camera intrisics.

    Follows the pattern found in:
    http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    of breaking the transform down into a perspective matrix followed by a
    transformation to NDC space.

    Args:
      focal_x: 1-D Tensor (or convertible value) with shape [batch_size]
        specifying the output focal length in the X dimension in pixels.
      focal_y: As focal_x, but for the Y dimension.
      center_offset_x: 1-D Tensor (or convertible value) with shape [batch_size]
        specifying the offset of the center of projection in the X dimension in
        pixels. A value of 0 puts an object at (0,0) in camera space in the center
        of the image. A positive value puts an object at (0,0) in the
        right half of the image, a negative value puts it in the left half.
      center_offset_y: As center_offset_x, but for the Y dimension. A positive
        value puts an object at (0,0) in camera space in the top half of the
        image, a negative value in the bottom half.
      near_clip: 1-D Tensor (or convertible value) with shape [batch_size]
        specifying near clipping plane distance.
      far_clip: 1-D Tensor (or convertible value) with shape [batch_size]
        specifying far clipping plane distance.
      image_width: int or float specifying the width of the camera's image
      image_height: int or float specifying the height of the camera's image

    Returns:
      A [batch_size, 4, 4] float32 Tensor that maps from right-handed points in
      camera space to left-handed points in clip space.
    """
    focal_x = to_tensor(focal_x)
    focal_y = to_tensor(focal_y)
    center_offset_x = to_tensor(center_offset_x)
    center_offset_y = to_tensor(center_offset_y)
    near_clip = to_tensor(near_clip)
    far_clip = to_tensor(far_clip)

    zeros = torch.zeros_like(focal_x)
    ones = torch.ones_like(focal_x)
    a = near_clip + far_clip
    b = near_clip * far_clip
    # pyformat: disable
    perspective_transform = torch.cat(
        [
            focal_x, zeros, -center_offset_x, zeros,
            zeros, focal_y, -center_offset_y, zeros,
            zeros, zeros, a, b,
            zeros, zeros, -ones, zeros
        ], dim=0)
    # pyformat: enable
    perspective_transform = torch.reshape(perspective_transform, [4, 4, -1])
    perspective_transform = perspective_transform.permute([2, 0, 1])

    ones = torch.ones_like(near_clip)
    left = ones * image_width * -0.5
    right = ones * image_width * 0.5
    bottom = ones * image_height * -0.5
    top = ones * image_height * 0.5
    ndc_transform = ortho(left, right, bottom, top, near_clip, far_clip)
    if debug:

        return torch.matmul(ndc_transform, perspective_transform), ndc_transform, perspective_transform
    else:
        return torch.matmul(ndc_transform, perspective_transform)


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
