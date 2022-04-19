from os.path import join
from glob import glob
import configs
from third_party.HRNet.config import config as hrnet_config
from third_party.HRNet.config import update_config
from third_party.HRNet import models as hrnet_models
import argparse
import torch


def parse_args(string=None):
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args(string)
    update_config(hrnet_config, args)
    return args


def get_paths():
    project_path = configs.project_path
    cfg_path = join(
        project_path, 'third_party/HRNet/config/seg_hrnet_ocr_w48_520x520_ohem_sgd_lr1e-3_wd1e-4_bs_16_epoch110.yaml')
    test_model_file = join(
        project_path, 'third_party/HRNet/pretrained/hrnet_ocr_cocostuff_3965_torch04.pth')
    pretrained_model_file = join(
        project_path, 'third_party/HRNet/pretrained/hrnetv2_w48_imagenet_pretrained.pth')

    return cfg_path, test_model_file, pretrained_model_file


def get_hrnet_segmentation_model():
    cfg_path, test_model_file, pretrained_model_file = get_paths()
    config_argument = ['--cfg', cfg_path, 'TEST.MODEL_FILE',
                       test_model_file, 'MODEL.PRETRAINED', pretrained_model_file]
    _ = parse_args(config_argument)
    model = eval('hrnet_models.'+hrnet_config.MODEL.NAME +
                 '.get_seg_model')(hrnet_config)

    model_state_file = hrnet_config.TEST.MODEL_FILE

    print('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
