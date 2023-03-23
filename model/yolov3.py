import torch
from torch import nn


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



if __name__ == '__main__':
    dummy_input = torch.randn(1, 3, 416, 416)
    conv_residual = ConvResidualBlock(in_channel=3, out_channel=64)
    out = conv_residual(dummy_input)
    print(out.shape)
