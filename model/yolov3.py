import torch
from torch import nn
from torch.nn import functional


class ConvBatchnormRelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=False):
        super(ConvBatchnormRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class ConvResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ConvResidualBlock, self).__init__()
        self.conv1 = ConvBatchnormRelu(in_channel, in_channel // 2, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBatchnormRelu(in_channel // 2, in_channel, kernel_size=3, stride=1, padding=1)

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
    def __init__(self):
        super(UpSampling, self).__init__()
        
    def forward(self, x):
        return functional.interpolate(x, scale_factor=2, mode='nearest')


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.trunk_52 = nn.Sequential(
            ConvBatchnormRelu(in_channel=3, out_channel=32, kernel_size=3, stride=1, padding=1),
            ConvBatchnormRelu(in_channel=32, out_channel=64, kernel_size=3, stride=2, padding=1),
            
            ConvResidualBlock(in_channel=64),
            DownSampling(in_channel=64, out_channel=128),
            
            ConvResidualBlock(in_channel=128), 
            ConvResidualBlock(in_channel=128),
            DownSampling(in_channel=128, out_channel=256),
            
            ConvResidualBlock(in_channel=256),
            ConvResidualBlock(in_channel=256),
            ConvResidualBlock(in_channel=256),
            ConvResidualBlock(in_channel=256),
            ConvResidualBlock(in_channel=256),
            ConvResidualBlock(in_channel=256),
            ConvResidualBlock(in_channel=256),
            ConvResidualBlock(in_channel=256),
        )
            
        self.trunk_26 = nn.Sequential(
            DownSampling(in_channel=256, out_channel=512),
            
            ConvResidualBlock(in_channel=512),
            ConvResidualBlock(in_channel=512),
            ConvResidualBlock(in_channel=512),
            ConvResidualBlock(in_channel=512),
            ConvResidualBlock(in_channel=512),
            ConvResidualBlock(in_channel=512),
            ConvResidualBlock(in_channel=512),
            ConvResidualBlock(in_channel=512),
        )
        
        self.trunk_13 = nn.Sequential(
            DownSampling(in_channel=512, out_channel=1024),
            
            ConvResidualBlock(in_channel=1024),
            ConvResidualBlock(in_channel=1024),
            ConvResidualBlock(in_channel=1024),
            ConvResidualBlock(in_channel=1024),
        )
            
        self.convset_13 = nn.Sequential(
            ConvSet(in_channel=1024, out_channel=512)
        )
        
        self.detection_13 = nn.Sequential(
            ConvBatchnormRelu(in_channel=512, out_channel=1024, kernel_size=3, stride=1, padding=1),
            ConvBatchnormRelu(in_channel=1024, out_channel=45, kernel_size=1, stride=1, padding=0)
        )
        
        self.up_26 = nn.Sequential(
            ConvBatchnormRelu(in_channel=512, out_channel=256, kernel_size=3, stride=1, padding=1),
            UpSampling()
        )
        
        self.convset_26 = nn.Sequential(
            ConvSet(in_channel=768, out_channel=256)
        )
        
        self.detection_26 = nn.Sequential(
            ConvBatchnormRelu(in_channel=256, out_channel=512, kernel_size=3, stride=1, padding=1),
            ConvBatchnormRelu(in_channel=512, out_channel=45, kernel_size=1, stride=1, padding=0)
        )
        
        self.up_52 = nn.Sequential(
            ConvBatchnormRelu(in_channel=256, out_channel=128, kernel_size=3, stride=1, padding=1),
            UpSampling()
        )
        
        self.convset_52 = nn.Sequential(
            ConvSet(in_channel=384, out_channel=128)
        )
        
        self.detection_52 = nn.Sequential(
            ConvBatchnormRelu(in_channel=128, out_channel=256, kernel_size=3, stride=1, padding=1),
            ConvBatchnormRelu(in_channel=256, out_channel=45, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        feature_52 = self.trunk_52(x)   # [1, 256, 52, 52]
        feature_26 = self.trunk_26(feature_52)  # [1, 512, 26, 26]
        feature_13 = self.trunk_13(feature_26)  # [1, 1024, 13, 13]
        
        conv_out_13 = self.convset_13(feature_13)   # [1, 512, 13, 13]
        out_13 = self.detection_13(conv_out_13)     # [1, 45, 13, 13]
        
        up_26 = self.up_26(conv_out_13)    # [1, 256, 26, 26]
        conv_out_26 = self.convset_26(torch.cat((up_26, feature_26), dim=1))    # [1, 256, 26, 26]
        out_26 = self.detection_26(conv_out_26)     # [1, 45, 26, 26]
        
        up_52 = self.up_52(conv_out_26)     # [1, 128, 52, 52]
        conv_out_52 = self.convset_52(torch.cat((up_52, feature_52), dim=1))    # [1, 128, 52, 52]
        out_52 = self.detection_52(conv_out_52)
        
        return out_13, out_26, out_52 
        
        

if __name__ == '__main__':
    dummy_input = torch.randn(1, 3, 416, 416)
    model = Darknet53()
    out = model(dummy_input)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
