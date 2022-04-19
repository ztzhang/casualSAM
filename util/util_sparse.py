import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

def Gumble_SoftMax(prob_map):
    # bernoli 
    B, C, H, W = prob_map.shape
    assert C == 1
    prob_map = torch.transpose(prob_map, 1, 3)  # BWHC
    prob_map = torch.cat((prob_map, 1 - prob_map), 3)
    prob_map = prob_map.view(-1,2)
    binary_map = F.gumbel_softmax(prob_map, hard=True,tau=0.01)
    binary_map = binary_map.view(B, W, H, 2)
    return binary_map[..., 0:1].transpose(1, 3)    


class STE(Function):
    @staticmethod
    def forward(ctx, prob_map, signal=None, threshold=0.1, clip_at=1):
        # ctx.clip_at = clip_at
        #prob_map_r = torch.zeros_like(prob_map)
        #prob_map_r[prob_map < threshold] = 0
        #prob_map_r[prob_map >= threshold] = 1
        #ctx.scale=False
        #if signal is not None:
        #    ctx.scale = True
        #    ctx.save_for_backward(signal)

        return (torch.sign(prob_map-threshold)+1)/2
        
    @staticmethod
    def backward(ctx, grad_output):
        # if ctx.scale:
        #    signal = ctx.saved_variables[0]
        #    return grad_output / (signal + 1e-4), None, None, None
        #else:
        return grad_output.clamp(-1, 1)/2, None, None, None    

class Sobel(nn.Module):
    def __init__(self):
        self.conv = F.conv2d
        self.weight = None
        super().__init__()

    def forward(self, tensor):
        if self.weight is None:
            self.weight = torch.zeros([2, 1, 3, 3], dtype=tensor.dtype, device=tensor.device)
            self.weight[0, 0, 0, 0] = -1
            self.weight[0, 0, 0, 1] = -2
            self.weight[0, 0, 0, 2] = -1
            self.weight[0, 0, 2, 0] = 1
            self.weight[0, 0, 2, 1] = 2
            self.weight[0, 0, 2, 2] = 1
            self.weight[1, 0, 0, 0] = 1
            self.weight[1, 0, 1, 0] = 2
            self.weight[1, 0, 2, 0] = 1
            self.weight[1, 0, 0, 2] = -1
            self.weight[1, 0, 1, 2] = -2
            self.weight[1, 0, 2, 2] = -1
        B, C, _, _ = tensor.shape
        Cout, Cin, _, _ = self.weight.shape
        assert Cout == 2
        if Cin != C:
            self.weight = self.weight.expand(-1, C, -1, -1)
        return self.conv(tensor, self.weight, padding=1)


# the ones that are not differentiable
def bin_dilation(bin_image):
    """binary dilation with 3x3 ones as structure element
    Arguments:
        bin_image {FloatTensor} -- binary image
    """
    return F.max_pool2d(bin_image, kernel_size = 3, stride = 1, padding = 1)


def bin_erosion(bin_image):
    """binary erosion with 3x3 ones as structure element   
    Arguments:
        bin_image {FloatTensor} -- binary image
    """
    return 1 - F.max_pool2d(1 - F.pad(bin_image, (1, 1, 1, 1)), kernel_size = 3, stride = 1, padding = 0)