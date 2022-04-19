import torch.multiprocessing as mp


def main_worker(local_rank, opt):
    """
    The main process for training.

    """
