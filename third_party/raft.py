
import sys
import argparse
import torch
from os.path import dirname, join
RAFT_PATH_ROOT = join(dirname(__file__), 'RAFT')
RAFT_PATH_CORE = join(RAFT_PATH_ROOT, 'core')
sys.path.append(RAFT_PATH_CORE)
from raft import RAFT  # nopep8
from utils.utils import InputPadder  # nopep8

# %%
# utility functions


def get_input_padder(shape):
    return InputPadder(shape, mode='sintel')


def load_RAFT():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision',
                        action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true',
                        help='use efficent correlation implementation')
    args = parser.parse_args(
        ['--model', join(RAFT_PATH_ROOT, 'models', 'raft-sintel.pth'), '--path', './'])
    net = RAFT(args)
    state_dict = torch.load(args.model)
    new_state_dict = {}
    for k in state_dict:
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = state_dict[k]
    net.load_state_dict(new_state_dict)
    return net.eval()
