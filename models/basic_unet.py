
import torch.nn as nn
import torch

def _normal_init(m, mean, std):
  """
  Helper function. Initialize model parameter with given mean and std.
  """
  if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    m.weight.data.normal_(mean, std)
    m.bias.data.zero_()

class Unet(nn.Module):
  # initializers
  def __init__(self, depth=False):
    super(Unet, self).__init__()
    
    self.in_channels = 4 if (depth==True) else 3
    
    # layer 1 - C64
    self.encConv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(4, 4), stride=2, padding=1)

    # layer 2 - C128
    self.encConv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=1)
    self.encBatchNorm2 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1) 

    # layer 3 - C256
    self.encConv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2, padding=1)
    self.encBatchNorm3 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1) 

    # layer 4 - C512
    self.encConv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=2, padding=1)
    self.encBatchNorm4 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1) 

    # layer 5 - C512
    self.encConv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=2, padding=1)
    self.encBatchNorm5 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1) 

    # layer 6 - C512
    self.encConv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=2, padding=1)
    self.encBatchNorm6 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1) 

    # layer 7 - C512
    self.encConv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=2, padding=1)
    self.encBatchNorm7 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1) 

    # layer 8 - C512
    self.encConv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=2, padding=1)
   
    # Layer 1 - C512
    self.decConv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4,4), stride=2, padding=1)
    self.decBatchNorm1 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1) 

    # Layer 2 - C512
    self.decConv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(4,4), stride=2, padding=1)
    self.decBatchNorm2 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1) 

    # Layer 3 - C512
    self.decConv3 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(4,4), stride=2, padding=1)
    self.decBatchNorm3 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1) 

    # Layer 4 - C512
    self.decConv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(4,4), stride=2, padding=1)
    self.decBatchNorm4 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1) 

    # Layer 5 - C256
    self.decConv5 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=(4,4), stride=2, padding=1)
    self.decBatchNorm5 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1) 

    # Layer 6 - C128
    self.decConv6 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(4,4), stride=2, padding=1)
    self.decBatchNorm6 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1) 

    # Layer 7 - C64
    self.decConv7 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(4,4), stride=2, padding=1)
    self.decBatchNorm7 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1) 

    # Layer 8 - C3
    self.decConv8 = nn.ConvTranspose2d(in_channels=128, out_channels=14, kernel_size=(4,4), stride=2, padding=1)

  # weight_init
  def weight_init(self, mean, std):
    for m in self._modules:
      _normal_init(self._modules[m], mean, std)

  # forward method
  def forward(self, input):

    e1_c = self.encConv1(input)
    
    e2_c = self.encConv2(nn.functional.leaky_relu(e1_c, 0.2))
    e2_bn = self.encBatchNorm2(e2_c)

    e3_c = self.encConv3(nn.functional.leaky_relu(e2_bn, 0.2))
    e3_bn = self.encBatchNorm3(e3_c)
    
    e4_c = self.encConv4(nn.functional.leaky_relu(e3_bn, 0.2))
    e4_bn = self.encBatchNorm4(e4_c)

    e5_c = self.encConv5(nn.functional.leaky_relu(e4_bn, 0.2))
    e5_bn = self.encBatchNorm5(e5_c)

    e6_c = self.encConv6(nn.functional.leaky_relu(e5_bn, 0.2))
    e6_bn = self.encBatchNorm6(e6_c)

    e7_c = self.encConv7(nn.functional.leaky_relu(e6_bn, 0.2))
    e7_bn = self.encBatchNorm7(e7_c)

    e8_c = self.encConv8(nn.functional.leaky_relu(e7_bn, 0.2))

    enc_output = nn.functional.leaky_relu(e8_c, 0.2)

    d1_c = self.decConv1(enc_output)
    d1_bn = self.decBatchNorm1(d1_c)
   
    d2_in = torch.cat([e7_bn, d1_bn], 1)
    
    d2_c = self.decConv2(nn.functional.relu(d2_in))
    d2_bn = self.decBatchNorm2(d2_c)

    d3_in = torch.cat([e6_bn, d2_bn], 1)

    d3_c = self.decConv3(nn.functional.relu(d3_in))
    d3_bn = self.decBatchNorm3(d3_c) 

    d4_in = torch.cat([e5_bn, d3_bn], 1)

    d4_c = self.decConv4(nn.functional.relu(d4_in))
    d4_bn = self.decBatchNorm4(d4_c) 

    d5_in = torch.cat([e4_bn, d4_bn], 1)

    d5_c = self.decConv5(nn.functional.relu(d5_in))
    d5_bn = self.decBatchNorm5(d5_c) 

    d6_in = torch.cat([e3_bn, d5_bn], 1)

    d6_c = self.decConv6(nn.functional.relu(d6_in))
    d6_bn = self.decBatchNorm6(d6_c) 

    d7_in = torch.cat([e2_bn, d6_bn], 1)

    d7_c = self.decConv7(nn.functional.relu(d7_in))
    d7_bn = self.decBatchNorm7(d7_c) 

    d8_in = torch.cat([e1_c, d7_bn], 1)

    d8_c = self.decConv8(nn.functional.relu(d8_in)) 

    output = torch.sigmoid(d8_c)

    return output
