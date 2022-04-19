import numpy as np
from util.util_colormap import heatmap_to_pseudo_color
KEYWORDS = ['depth', 'edgegradient', 'flow', 'img',
            'rgb', 'image', 'edge', 'contour', 'softmask']


def detach_to_cpu(tensor):
    if type(tensor) == np.ndarray:
        return tensor
    else:
        if tensor.requires_grad:
            tensor.requires_grad = False
        tensor = tensor.cpu()
    return tensor.numpy()


def gen_vid(image_path_pattern, video_path):
    from subprocess import call
    gen_vid_command = 'ffmpeg -nostats -loglevel 0 -y -r 30 -i {img_template} -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {video_path}.mp4 > /dev/null'
    vid_to_gif_command = 'ffmpeg -nostats -loglevel 0 -y -i {video_path}.mp4 -filter_complex "[0:v] split [a][b];[a] palettegen=stats_mode=single [p];[b][p] paletteuse=new=1" {video_path}.gif'
    call(gen_vid_command.format(img_template=image_path_pattern,
         video_path=video_path), shell=True)
    call(vid_to_gif_command.format(video_path=video_path), shell=True)


class Converter:
    def __init__(self):
        pass

    @staticmethod
    def depth2img(tensor, normalize=True, disparity=True, eps=1e-6, **kargs):
        t = detach_to_cpu(tensor)
        assert len(t.shape) == 4
        assert t.shape[1] == 1
        t = 1 / (t + eps)
        # if normalize:
        max_v = np.max(t, axis=(2, 3), keepdims=True)
        min_v = np.min(t, axis=(2, 3), keepdims=True)
        t = (t - min_v) / (max_v - min_v + eps)
        #    return t
        # else:
        #    return t
        cs = []
        for b in range(t.shape[0]):
            c = heatmap_to_pseudo_color(t[b, 0, ...])
            cs.append(c[None, ...])
        cs = np.concatenate(cs, axis=0)
        cs = np.transpose(cs, [0, 3, 1, 2])
        return cs

    @staticmethod
    def edge2img(tensor, normalize=True, eps=1e-6, **kargs):
        t = detach_to_cpu(tensor)
        if np.max(t) > 1 or np.min(t) < 0:
            t = 1 / (1 + np.exp(-t))
        assert len(t.shape) == 4
        assert t.shape[1] == 1
        return t

    @staticmethod
    def image2img(tensor, **kargs):
        return Converter.img2img(tensor)

    @staticmethod
    def softmask2img(tensor, **kargs):
        t = detach_to_cpu(tensor)  # [:, None, ...]
        # t = #detach_to_cpu(tensor)
        return t

    @staticmethod
    def scenef2img(tensor, **kargs):
        t = detach_to_cpu(tensor.squeeze(3))
        assert len(t.shape) == 4
        return np.linalg.norm(t, ord=1, axis=-1, keepdims=True)

    @staticmethod
    def rgb2img(tensor, **kargs):
        return Converter.img2img(tensor)

    @staticmethod
    def img2img(tensor, **kargs):
        t = detach_to_cpu(tensor)
        if np.min(t) < -0.1:
            t = (t + 1) / 2
        elif np.max(t) > 1.5:
            t = t / 255
        return t

    @staticmethod
    def edgegradient2img(tensor, **kargs):
        t = detach_to_cpu(tensor)
        mag = np.max(abs(t))
        positive = np.where(t > 0, t, 0)
        positive /= mag
        negative = np.where(t < 0, abs(t), 0)
        negative /= mag
        rgb = np.concatenate(
            (positive, negative, np.zeros(negative.shape)), axis=1)
        return rgb

    @staticmethod
    def flow2img(tensor, **kargs):
        t = detach_to_cpu(tensor)
        return t


def convert2rgb(tensor, key, **kargs):
    found = False
    for k in KEYWORDS:
        if k in key:
            convert = getattr(Converter, k + '2img')
            found = True
            break
    if not found:
        return None
    else:
        return convert(tensor, **kargs)


def is_key_image(key):
    """check if the given key correspondes to images

    Arguments:
        key {str} -- key of a data pack

    Returns:
        bool -- [True if the given key correspondes to an image]
    """

    for k in KEYWORDS:
        if k in key:
            return True
    return False


def parse_key(key):
    rkey = None
    found = False
    mode = None
    for k in KEYWORDS:
        if k in key:
            rkey = k
            found = True
            break
    if 'pred' in key:
        mode = 'pred'
    elif 'gt' in key:
        mode = 'gt'
    if not found:
        return None, None
    else:
        return rkey, mode
