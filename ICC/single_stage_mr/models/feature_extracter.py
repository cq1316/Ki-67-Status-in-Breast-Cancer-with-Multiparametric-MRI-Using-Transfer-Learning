from torchvision.models.vgg import vgg16_bn
import torch.nn as nn
import torch


class FeatureExtracter():
    def __init__(self, m):
        m.register_forward_hook(self.hook_fn)

    def hook_fn(self, moudle, input, output):
        self.feature = output


class PoolVgg(nn.Module):
    def __init__(self):
        super(PoolVgg, self).__init__()
        self.vgg = nn.Sequential(*list(vgg16_bn(pretrained=True).features.children()))
        self.pool_features = [FeatureExtracter(self.vgg[6]),
                              FeatureExtracter(self.vgg[13]),
                              FeatureExtracter(self.vgg[23]),
                              FeatureExtracter(self.vgg[33])]
        self.avg1 = nn.AdaptiveAvgPool2d(1)
        self.avg2 = nn.AdaptiveAvgPool2d(1)
        self.avg3 = nn.AdaptiveAvgPool2d(1)
        self.avg4 = nn.AdaptiveAvgPool2d(1)
        self.avg5 = nn.AdaptiveAvgPool2d(1)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input):
        batch_size = input.size(0)
        b5_out = self.vgg(input)
        b1_out = self.pool_features[0].feature
        b2_out = self.pool_features[1].feature
        b3_out = self.pool_features[2].feature
        b4_out = self.pool_features[3].feature
        b1_avg = self.avg1(b1_out).view(batch_size, -1)
        b2_avg = self.avg2(b2_out).view(batch_size, -1)
        b3_avg = self.avg3(b3_out).view(batch_size, -1)
        b4_avg = self.avg4(b4_out).view(batch_size, -1)
        b5_avg = self.avg5(b5_out).view(batch_size, -1)
        vgg_out = torch.cat([b1_avg, b2_avg, b3_avg, b4_avg, b5_avg], dim=1)
        return vgg_out
