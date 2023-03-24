import torch
from torch import nn
from torch.nn import functional


class ConvBatchnormRelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBatchnormRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class ConvResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvResidualBlock, self).__init__()
        self.conv1 = ConvBatchnormRelu(in_channel, out_channel // 2, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBatchnormRelu(out_channel // 2, in_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x
    

class ConvSet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvSet, self).__init__()
        self.layer = nn.Sequential(
            ConvBatchnormRelu(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            ConvBatchnormRelu(out_channel, in_channel, kernel_size=3, stride=1, padding=1),
            ConvBatchnormRelu(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            ConvBatchnormRelu(out_channel, in_channel, kernel_size=3, stride=1, padding=1),
            ConvBatchnormRelu(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
        )
    
    def forward(self, x):
        return self.layer(x)
    

class DownSampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSampling, self).__init__()
        self.layer = nn.Sequential(
            ConvBatchnormRelu(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.layer(x)
    


class UpSampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSampling, self).__init__()
        
    def forward(self, x):
        return functional.interpolate(x, scale_factor=2, mode='nearest')


class ConvSet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvSet).__init__()
        self.


if __name__ == '__main__':
    dummy_input = torch.randn(1, 3, 416, 416)
    conv_residual = ConvResidualBlock(in_channel=3, out_channel=64)
    out = conv_residual(dummy_input)
    print(out.shape)
