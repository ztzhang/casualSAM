import numpy as np
from os.path import join
from skimage.transform import resize as imresize
from .util_3dvideo import depth_to_points_Rt


def process_for_pointcloud_video(img, disp, mask, R, t, K, downsample=2, rot_mat=None):
    if rot_mat is None:
        rot_mat = np.eye(3)
    H, W, C = img.shape
    dh = H//downsample
    dw = W//downsample
    img = imresize(img, (dh, dw), preserve_range=True)
    disp = imresize(disp, (dh, dw), preserve_range=True)
    mask_motion = imresize(mask, (dh, dw), preserve_range=True)
    mask_motion = np.where(mask_motion > 0.99, 1, 0)
    K[:2, :] = K[:2, :]/downsample
    pts = depth_to_points_Rt(1/(disp+1e-6), R, t, K)
    mask = mask_motion.flatten()
    rgb = img.reshape([-1, 3])
    pts = rot_mat.dot(pts.T).T
    return pts, rgb, mask, R, t, K, dh, dw


class PointCloudVideoWebpage():
    global_code_template = '''
    var camera_geometry_list = [];
    var point_geometry_list = [];
    const pointSize = 0.002;
    let buffer_geometry, positions, colors, geometry, wireframe, line, wire_geometry, material;
    const line_indices = [0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 1, 4, 2, 4, 3, 4];
    let camera_positions, camera_geometry, line_segments, camera_material;
    '''

    point_cloud_code_template = '''
    buffer_geometry = new THREE.BufferGeometry();
    positions = new Float32Array([{xyz}]);
    colors = new Float32Array([{rgb}]);
    buffer_geometry.attributes.position = new THREE.BufferAttribute( positions, 3 );
    buffer_geometry.attributes.color = new THREE.BufferAttribute( colors, 3 );
    material = new THREE.PointsMaterial( {{ size: pointSize, vertexColors: true }} );
    geometry = new THREE.Points( buffer_geometry, material );
    point_geometry_list.push(geometry);
    '''

    camera_wireframe_code_template = '''
    camera_positions = new Float32Array([{xyz}]);
    camera_geometry = new THREE.BufferGeometry();
    camera_geometry.addAttribute('position', new THREE.BufferAttribute(camera_positions, 3));
    camera_geometry.setIndex( line_indices );
    camera_material = new THREE.LineBasicMaterial({{color: 0x5c5c5c}});
    line_segments = new THREE.LineSegments(camera_geometry, camera_material);
    camera_geometry_list.push(line_segments);
    '''

    video3d_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=windows-1252">
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
        <meta http-equiv="Pragma" content="no-cache">
        <meta http-equiv="Expires" content="0">
        <title>3D video</title>
        <script src="http://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="http://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="http://ztzhang.info/assets/three_js/OrbitControls.js"></script>
        <script src="http://vision38.csail.mit.edu/data/vision/billf/scratch/ztzhang/layered-video/src_code/dev/js_based_3d_video/TrackballControls.js"></script>
        <script>{geometry_string}</script>
        <style>
            #c {{
                position: fixed;
                left: 0px;
                width: 100%;
                height: 100%;
                z-index: -1
            }}
        </style>
    </head>
    <body>
        <script type="module" src="http://vision38.csail.mit.edu/data/vision/billf/scratch/ztzhang/layered-video/src_code/dev/js_based_3d_video/3d_video_viewer.js"></script>
    </body>

    </html>
    """

    def __init__(self):
        self.html_template = self.video3d_template
        self.mesh_snippet = self.global_code_template
        self.camera_snippet = ''
        self.camera_registerd = False

    def register_camera_intrinsics(self, H, W, K, viz_depth=0.01):
        coord_x = [0, 0, W-1, W-1]
        coord_y = [0, H-1, H-1, 0]
        coord_z = [1, 1, 1, 1]
        coord = np.asarray([coord_x, coord_y, coord_z])
        coord_3D = viz_depth*np.linalg.inv(K).dot(coord)
        self.coord_3D = np.concatenate([coord_3D, np.zeros((3, 1))], axis=1)
        self.camera_registerd = True

    def add_pointcloud(self, pts, rgb, mask):
        if mask is None:
            mask = np.ones_like(pts[:, 0])
        xyz_string = []
        rgb_string = []
        for i in range(pts.shape[0]):
            if mask[i] == 0:
                continue
            xyz_string.append(
                f'{pts[i, 0]:.4f},{pts[i, 1]:.4f},{pts[i, 2]:.4f},')
            rgb_string.append(
                f'{rgb[i, 0]:.4f},{rgb[i, 1]:.4f},{rgb[i, 2]:.4f},')
        xyz_string = ''.join(xyz_string)
        rgb_string = ''.join(rgb_string)
        self.mesh_snippet += self.point_cloud_code_template.format(
            xyz=xyz_string, rgb=rgb_string)

    def add_camera(self, R, t, rot_mat=None):
        if rot_mat is None:
            rot_mat = np.eye(3)
        camera_xyz = R@self.coord_3D + t
        camera_xyz = rot_mat@camera_xyz
        camera_xyz_string = ''
        for i in range(camera_xyz.shape[1]):
            camera_xyz_string += f'{camera_xyz[0, i]:.4f},{camera_xyz[1, i]:.4f},{camera_xyz[2, i]:.4f},'
        self.camera_snippet += self.camera_wireframe_code_template.format(
            xyz=camera_xyz_string)

    def save(self, output_path, name=None):
        if name is not None:
            output_path = join(output_path, name)
        with open(output_path, 'w') as f:
            f.write(self.html_template.format(
                geometry_string=self.mesh_snippet+self.camera_snippet))
