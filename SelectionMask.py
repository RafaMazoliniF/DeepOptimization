import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class SelectionMask(nn.Module):    
    def __init__(self, shape, pre_mask=None) -> None:
        super().__init__()
        if pre_mask == None:
            tensor = torch.randn(shape, requires_grad=True)
            self.mask = nn.Parameter(nn.init.normal_(tensor=tensor, mean=2, std=0.01))
        else:
            self.mask = nn.Parameter(pre_mask)
        
    def forward(self, x):
        """
        Applies the binarized mask to an image keeping the grandient information
        """
        sig = torch.sigmoid(self.mask)
        bin_mask = (sig > 0.5).float()
        diff_mask = bin_mask + (sig - sig.detach())
        return x * diff_mask


def mask_l1_loss(mask: SelectionMask):
    """
    Returns the percentage of the mask with a value of 1
    """
    sig = torch.sigmoid(mask.mask)
    bin_mask = (sig > 0.5).float()
    diff_mask = bin_mask + (sig - sig.detach())
    return diff_mask.sum() / diff_mask.numel()