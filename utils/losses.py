import torch
import torch.nn as nn
from monai.losses import DiceLoss as MONAIDiceLoss
from monai.losses import SoftclDiceLoss


# Use MONAI's DiceLoss
class DiceLoss(MONAIDiceLoss):
    """
    MONAI Dice Loss for segmentation
    https://github.com/Project-MONAI/MONAI/blob/dev/monai/losses/dice.py
    """
    def __init__(self, sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5):
        super(DiceLoss, self).__init__(
            sigmoid=sigmoid,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            squared_pred=False,
            reduction='mean'
        )


# Use MONAI's SoftclDiceLoss
class SoftclDiceLossWrapper(SoftclDiceLoss):
    """
    MONAI Soft centerline Dice Loss for tubular structure segmentation
    https://github.com/Project-MONAI/MONAI/blob/dev/monai/losses/cldice.py
    Note: SoftclDiceLoss expects logits (applies sigmoid internally)
    """
    def __init__(self, iter_=3, smooth=1.0):
        super(SoftclDiceLossWrapper, self).__init__(
            iter_=iter_,
            smooth=smooth
        )


class CombinedLoss(nn.Module):
    """Combined Dice, clDice, and BCE Loss"""
    
    def __init__(self, dice_weight=0.4, cldice_weight=0.3, bce_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.cldice_loss = SoftclDiceLossWrapper(iter_=3, smooth=1.0)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        cldice = self.cldice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.cldice_weight * cldice + self.bce_weight * bce


def calculate_dice_score(pred, target, threshold=0.5):
    """Calculate Dice score for evaluation"""
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-8)
    
    return dice.item()


def calculate_cldice_score(pred, target, threshold=0.5):
    """
    Calculate centerline Dice (clDice) score for evaluation using MONAI
    """
    try:
        # Use MONAI's soft clDice for evaluation
        cldice_loss_fn = SoftclDiceLossWrapper(iter_=3, smooth=1.0)
        
        # Calculate loss (lower is better)
        loss = cldice_loss_fn(pred, target)
        
        # Convert loss to score (1 - loss, higher is better)
        cldice_score = 1.0 - loss.item()
        
        return max(0.0, min(1.0, cldice_score))  # Clamp to [0, 1]
    except Exception as e:
        print(f"Warning: clDice calculation failed: {e}")
        return 0.0
