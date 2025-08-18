import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class SelectionMask(nn.Module):
    def __init__(self, shape, pre_mask=None) -> None:
        super().__init__()
        
        tensor = torch.randn(shape, requires_grad=True)
        
        if pre_mask == None:
            self.mask = nn.Parameter(nn.init.normal_(tensor=tensor, mean=2, std=1))
        else:
            self.mask = nn.Parameter(pre_mask)
        
    def forward(self, x):
        sig = torch.sigmoid(self.mask)
        return x * (torch.abs(sig) > 0.5).float()
    


def mask_l1_loss(mask: SelectionMask):
    sig = torch.sigmoid(mask.mask)
    return sig.sum() / sig.numel()