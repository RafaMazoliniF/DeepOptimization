import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class SelectionMask(nn.Module):
    """
    A learnable binary mask module that applies element-wise selection to input tensors.
    
    The mask uses a sigmoid activation followed by thresholding to create binary selections.
    The underlying parameters are continuous and differentiable, allowing gradient-based
    optimization while producing discrete binary masks during forward passes.
    
    Args:
        shape (tuple): Shape of the mask tensor to be created
        pre_mask (torch.Tensor, optional): Pre-initialized mask parameters. If None,
            mask is initialized with normal distribution (mean=2, std=1). Defaults to None.
    
    Attributes:
        mask (nn.Parameter): Learnable parameter tensor that gets converted to binary mask
    """
    
    def __init__(self, shape, pre_mask=None) -> None:
        super().__init__()
        
        tensor = torch.randn(shape, requires_grad=True)
        
        if pre_mask == None:
            self.mask = nn.Parameter(nn.init.normal_(tensor=tensor, mean=2, std=1))
        else:
            self.mask = nn.Parameter(pre_mask)
        
    def forward(self, x):
        """
        Apply the binary selection mask to the input tensor.
        
        The mask parameters are passed through sigmoid activation and then thresholded
        at 0.5 to create binary selections. Elements where the sigmoid output has
        absolute value > 0.5 are kept, others are zeroed out.
        
        Args:
            x (torch.Tensor): Input tensor to be masked
            
        Returns:
            torch.Tensor: Input tensor with binary mask applied element-wise
        """
        sig = torch.sigmoid(self.mask)
        return x * (torch.abs(sig) > 0.5).float()


def mask_l1_loss(mask: SelectionMask):
    """
    Compute L1 regularization loss for the binary mask to encourage sparsity.
    
    This loss function calculates the fraction of active (non-zero) elements in the
    binary mask. It can be used as a regularization term to encourage the model
    to use fewer mask elements, promoting sparsity in the selection.
    
    Args:
        mask (SelectionMask): The SelectionMask module to compute loss for
        
    Returns:
        torch.Tensor: Scalar tensor representing the fraction of active mask elements
            (values range from 0.0 to 1.0, where 0.0 means all elements are masked out
            and 1.0 means all elements are active)
    """
    sig = torch.sigmoid(mask.mask)
    bin_mask = (torch.abs(sig) > 0.5).float()
    return bin_mask.sum() / bin_mask.numel()