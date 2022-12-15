from utils import DiceLoss, DiceLoss2
import torch

# create two random tensors of size (16, 13, 512, 512) 

pred = torch.rand(16, 13, 512, 512)
target = torch.rand(16, 13, 512, 512)

loss1 = DiceLoss(pred, target)
loss2 = DiceLoss2(pred, target)

print(loss1, loss2)