import torch.nn as nn
import torch

class FirstBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=14):
        super(FirstBlock, self).__init__()
        self.threeXthree = nn.Conv2d(in_channels = in_channels, out_channels = 16 - in_channels, kernel_size = 3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride=2, padding = 0)
        self.PReLU = nn.PReLU(num_parameters= 16)
        self.BatchNorm2D = nn.BatchNorm2d(num_features = 16)
        

    def forward(self, input):
        three_X_three = self.threeXthree(input)
        max_pool = self.max_pool(input)
        result = torch.cat((three_X_three, max_pool), 1)
        result = self.BatchNorm2D(result)
        result = self.PReLU(result)
        return result

class bottleneck_sample(nn.Module):
    def __init__(self, in_channels, out_channels, p, upsample = False, downsample = False):
        super(bottleneck_sample, self).__init__()
        bneck_channels = in_channels//4
        projection_kernel_size = 0
        self.upsample = upsample
        self.downsample = downsample
        if downsample:
            projection_kernel_size = 2
        elif upsample:
            projection_kernel_size = 1
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, bneck_channels, kernel_size = projection_kernel_size, bias = False),
            nn.BatchNorm2d(bneck_channels),
            nn.PReLU(),
        )

        if downsample:
            self.middle = nn.Sequential(
                nn.Conv2d(bneck_channels, bneck_channels, kernel_size = 3, stride = 2, padding = 1, bias = False),
                nn.BatchNorm2d(bneck_channels),
                nn.PReLU()
            )
        elif upsample:
            self.middle = nn.Sequential(
                nn.ConvTranspose2d(bneck_channels, bneck_channels, kernel_size = 2, stride = 2, padding = 1, output_padding = 1, bias = False),
                nn.BatchNorm2d(bneck_channels),
                nn.PReLU()
            )
        self.expansion = nn.Sequential(
                nn.Conv2d(bneck_channels, out_channels, kernel_size = 1, bias = False),
                nn.BatchNorm2d(out_channels),
                nn.PReLU()
        )

        self.padding = out_channels - in_channels
        self.finalPReLU = nn.PReLU()
        self.dp = nn.Dropout2d(p = p)

        if downsample:
            self.pool = nn.MaxPool2d(kernel_size = 2, stride=2, return_indices = True)
        elif upsample:
            self.pool = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
            self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn_spatial = nn.BatchNorm2d(out_channels)

    def forward(self, input, indices=None, size=None):
        right_result = self.projection(input)
        if self.upsample:
            right_result = self.middle(right_result)
        else:
            right_result = self.middle(right_result)
        right_result = self.expansion(right_result)
        right_result = self.dp(right_result)
        if self.downsample:
            left_result, indices = self.pool(input)
        elif self.upsample:
            hold = input
           
            left_result = self.spatial_conv(hold)
            left_result = self.bn_spatial(left_result)
            left_result = self.pool(left_result, indices, size)

        if self.downsample and self.padding > 0:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pad = torch.zeros((left_result.size(0), self.padding, left_result.size(2), left_result.size(3))).to(device=device)
            left_result = torch.cat((left_result,pad), 1)
        if left_result.size() != right_result.size():
            pads = (left_result.size(3) - right_result.size(3), 0, left_result.size(2) - right_result.size(2), 0)
            right_result = nn.functional.pad(right_result, pads, "constant", 0)
        result = right_result + left_result

        result = self.finalPReLU(result)
        if self.downsample:
            return result, indices
        else:
            return result
    
class bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, p, asymmetry=False, dilation=1, padding=0):
        super(bottleneck, self).__init__()
        bneck_channels = in_channels//4
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, bneck_channels, kernel_size = 1, bias=False),
            nn.BatchNorm2d(bneck_channels),
            nn.PReLU()
        )

        if asymmetry:
            self.middle = nn.Sequential(
                nn.Conv2d(bneck_channels, bneck_channels, kernel_size = (kernel_size,1), padding = (padding,0), dilation=dilation, bias=False),
                nn.BatchNorm2d(bneck_channels),
                nn.PReLU(),
                nn.Conv2d(bneck_channels, bneck_channels, kernel_size = (1,kernel_size), padding = (0,padding), dilation=dilation, bias=False),
                nn.BatchNorm2d(bneck_channels),
                nn.PReLU(),
            )
        else:
            self.middle = nn.Sequential(
                nn.Conv2d(bneck_channels, bneck_channels, kernel_size = kernel_size, padding = padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(bneck_channels),
                nn.PReLU(),
            )

        self.expansion = nn.Sequential(
            nn.Conv2d(bneck_channels, out_channels, kernel_size = 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        
        self.dp = nn.Dropout2d(p=p)
        self.finalPReLU = nn.PReLU()
    def forward(self, input):
        result = self.projection(input)
        result = self.middle(result)
        result = self.expansion(result)
        result = self.dp(result)
        result = result + input
        result = self.finalPReLU(result)
        return result

class enet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(enet, self).__init__()
        self.firstBlock = FirstBlock(in_channels)

        # Stage One
        self.bneck_10 = bottleneck_sample(in_channels=16, out_channels=64, p=.01, downsample = True)
        self.bneck_11 = bottleneck(in_channels=64, out_channels=64, kernel_size = 3, p=.01, padding = 1)
        self.bneck_12 = bottleneck(in_channels=64, out_channels=64, kernel_size = 3, p=.01, padding = 1)
        self.bneck_13 = bottleneck(in_channels=64, out_channels=64, kernel_size = 3, p=.01, padding = 1)
        self.bneck_14 = bottleneck(in_channels=64, out_channels=64, kernel_size = 3, p=.01, padding = 1)

        # Stage Two
        self.bneck_20 = bottleneck_sample(in_channels=64, out_channels=128, p=.1, downsample = True)
        self.bneck_21 = bottleneck(in_channels=128, out_channels=128, kernel_size = 3, p=.1, asymmetry=False, dilation=1, padding=1)
        self.bneck_22 = bottleneck(in_channels=128, out_channels=128, kernel_size = 3, p=.1, asymmetry=False, dilation=2, padding=2)
        self.bneck_23 = bottleneck(in_channels=128, out_channels=128, kernel_size = 5, p=.1, asymmetry=True, dilation=1, padding=2)
        self.bneck_24 = bottleneck(in_channels=128, out_channels=128, kernel_size = 3, p=.1, asymmetry=False, dilation=4, padding=4)
        self.bneck_25 = bottleneck(in_channels=128, out_channels=128, kernel_size = 3, p=.1, asymmetry=False, dilation=1, padding=1)
        self.bneck_26 = bottleneck(in_channels=128, out_channels=128, kernel_size = 3, p=.1, asymmetry=False, dilation=8, padding=8)
        self.bneck_27 = bottleneck(in_channels=128, out_channels=128, kernel_size = 5, p=.1, asymmetry=True, dilation=1, padding=2)
        self.bneck_28 = bottleneck(in_channels=128, out_channels=128, kernel_size = 3, p=.1, asymmetry=False, dilation=16, padding=16)

        #Stage Three
        self.bneck_30 = bottleneck(in_channels=128, out_channels=128, kernel_size = 3, p=.1, asymmetry=False, dilation=1, padding=1)
        self.bneck_31 = bottleneck(in_channels=128, out_channels=128, kernel_size = 3, p=.1, asymmetry=False, dilation=2, padding=2)
        self.bneck_32 = bottleneck(in_channels=128, out_channels=128, kernel_size = 5, p=.1, asymmetry=True, dilation=1, padding=2)
        self.bneck_33 = bottleneck(in_channels=128, out_channels=128, kernel_size = 3, p=.1, asymmetry=False, dilation=4, padding=4)
        self.bneck_34 = bottleneck(in_channels=128, out_channels=128, kernel_size = 3, p=.1, asymmetry=False, dilation=1, padding=1)
        self.bneck_35 = bottleneck(in_channels=128, out_channels=128, kernel_size = 3, p=.1, asymmetry=False, dilation=8, padding=8)
        self.bneck_36 = bottleneck(in_channels=128, out_channels=128, kernel_size = 5, p=.1, asymmetry=True, dilation=1, padding=2)
        self.bneck_37 = bottleneck(in_channels=128, out_channels=128, kernel_size = 3, p=.1, asymmetry=False, dilation=16, padding=16)

        # Stage Four
        self.bneck_40 = bottleneck_sample(in_channels = 128, out_channels = 64, p=.01, upsample = True)
        self.bneck_41 = bottleneck(in_channels=64, out_channels=64, kernel_size = 3, p=.1, asymmetry=False, dilation=1, padding=1)
        self.bneck_42 = bottleneck(in_channels=64, out_channels=64, kernel_size = 3, p=.1, asymmetry=False, dilation=1, padding=1)

        #Stage Five
        self.bneck_50 = bottleneck_sample(in_channels = 64, out_channels = 16, p=.01, upsample = True)
        self.bneck_51 = bottleneck(in_channels=16, out_channels=16, kernel_size = 3, p=.1, asymmetry=False, dilation=1, padding=1)

        self.fc = nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, input):
        #print("Second ENet")
        res = self.firstBlock(input)
        #Stage One
        size1 = res.size()
        res, indices1 = self.bneck_10(res)
        res = self.bneck_11(res)
        res = self.bneck_12(res)
        res = self.bneck_13(res)
        res = self.bneck_14(res)

        #Stage Two
        size2 = res.size()
        res, indices2 = self.bneck_20(res)
        res = self.bneck_21(res)
        res = self.bneck_22(res)
        res = self.bneck_23(res)
        res = self.bneck_24(res)
        res = self.bneck_25(res)
        res = self.bneck_26(res)
        res = self.bneck_27(res)
        res = self.bneck_28(res)

        #Stage Three
        res = self.bneck_30(res)
        res = self.bneck_31(res)
        res = self.bneck_32(res)
        res = self.bneck_33(res)
        res = self.bneck_34(res)
        res = self.bneck_35(res)
        res = self.bneck_36(res)
        res = self.bneck_37(res)

        #Stage 4
        res = self.bneck_40(res, indices=indices2, size = size2)
        res = self.bneck_41(res)
        res = self.bneck_42(res)

        #Stage 5
        res = self.bneck_50(res, indices = indices1, size = size1)
        res = self.bneck_51(res)

        res = self.fc(res)

        return res




