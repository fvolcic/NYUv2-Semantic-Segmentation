
import torch
import numpy as np 
import matplotlib.pyplot as plt
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_to_one_hot(labels, classes=14, scale_values=True):
    """
    Convert a segmentation image to one hot encoding

    @params
    labels: (batch_size, height, width) or (batch_size, 1, height, width)
    classes: number of classes
    scale_values: if True, labels is multiplied by classes then rounded to the nearest integer before converting to one hot encoding.

    @note
    labels must be integers between 0 and classes-1 unless scale_values is True. 
    If scale_values is True, labels must be between 0 and 1, 
    and will be multiplied by classes before being converted to one hot encoding.

    @return
    one_hot_images: (batch_size, num_classes, height, width)
    """

    if len(labels.shape) == 4:
        labels = labels.squeeze(1)
    
    if scale_values:
        labels = torch.round(labels * (classes-1))
    
    one_hot = torch.nn.functional.one_hot(labels.long(), num_classes=classes)
    one_hot = torch.permute(one_hot, [0, 3, 1, 2]).float()

    return one_hot

def convert_to_segmentation(one_hot_images):
    """
    Convert a one hot encoding image to segmentation image

    @params
    one_hot_images: (batch_size, num_classes, height, width)

    @return
    segmentation_images: (batch_size, 1, height, width)
    """
    return torch.argmax(one_hot_images, dim=1)

def meanIoU(pred, target, classes):
    """
    Give the mean IoU of the image and the mask
    
    @params
        img: (batch_size, classes, height, width)
        mask: (batch_size, classes, height, width)
        classes: number of classes
        
    @note it is required that the values in img and mask and intergers between 0 and classes-1 

    @returns
        mean IoU
    """
    
    # convert pred to segmentation and back to one hot encoding
    pred = convert_to_one_hot(convert_to_segmentation(pred), classes=classes, scale_values=False)
    
    intersection = torch.sum(pred * target, dim=(2, 3))
    union = torch.sum(pred, dim=(2, 3)) + torch.sum(target, dim=(2, 3)) - intersection
    iou = (intersection + 1e-8) / (union + 1e-8)
    iou = torch.mean(iou, dim=1)
    return torch.mean(iou)

class DiceLoss(nn.Module):
    """
    An implementation of the dice loss function.
    """
    def __init__(self, num_classes=14):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        intersection = torch.sum(pred * target, dim=(1, 2, 3))
        union = torch.sum(pred, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))
        dice = (2 * intersection ) / (union + 1e-8)
        dice = torch.mean(1 - dice)
        return dice 

def compute_pixel_accuracy(pred, y):
    """
    pred - torch tensor of shape (N, C, H, W)
    y - torch tensor of shape (N, C, H, W)
    """

    pred_mask = convert_to_segmentation(pred)
    y_mask = convert_to_segmentation(y)

    correct = (pred_mask == y_mask).sum()
    total = pred_mask.shape[0] * pred_mask.shape[1] * pred_mask.shape[2]

    return correct / total