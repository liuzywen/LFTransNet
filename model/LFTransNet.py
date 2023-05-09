import torch
import torch.nn as nn
from thop import profile, clever_format

from bakebone.pvtv2 import pvt_v2_b2
from transformer_decoder import transfmrerDecoder
import torch.nn.functional as F
from model.MultiScaleAttention import Block

from fvcore.nn import FlopCountAnalysis, parameter_count_table



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x




class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x




class decoder1(nn.Module):
    def __init__(self, channels):
        super(decoder1, self).__init__()
        self.convf = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(2*channels, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(3*channels, channels, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(4*channels, channels, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear'),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
    def forward(self, x3, x4, x5, x6):
        x3 = self.upsample(x3)
        x4 = self.conv4(torch.cat((x3, x4), dim=1))
        x3 = self.upsample(x3)
        x4 = self.upsample(x4)
        x5 = self.conv5(torch.cat((x3, x4, x5), dim=1))
        x5 = self.upsample(x5)
        x4 = self.upsample(x4)
        x3 = self.upsample(x3)
        x6 = self.conv6(torch.cat((x3, x4, x5, x6), dim=1))

        return x6


class decoder2(nn.Module):
    def __init__(self, channels):
        super(decoder2, self).__init__()
        self.convf = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(2*channels, channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(3*channels, channels, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(4*channels, channels, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)
    def forward(self, x3, x4, x5, x6):
        x3 = self.upsample(x3)
        x4 = self.conv4(torch.cat((x3, x4), dim=1))
        x3 = self.upsample(x3)
        x4 = self.upsample(x4)
        x5 = self.conv5(torch.cat((x3, x4, x5), dim=1))
        x5 = self.upsample(x5)
        x4 = self.upsample(x4)
        x3 = self.upsample(x3)
        x6 = self.conv6(torch.cat((x3, x4, x5, x6), dim=1))

        return x6

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class Rgb_guide_sa(nn.Module):
    def __init__(self):
        super(Rgb_guide_sa, self).__init__()
        self.SA = SpatialAttention()


    def forward(self, rgb, focal, focal2):
        rgb_sa = self.SA(rgb)
        focal_sa = torch.mul(rgb_sa, focal)
        focal2 = F.interpolate(focal2, scale_factor=2, mode='bilinear', align_corners=False)
        focal = focal_sa + focal +focal2

        return focal




class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        #focal
        self.focal_encoder = pvt_v2_b2()
        self.rfb4 = RFB_modified(512, 32)
        self.rfb3 = RFB_modified(320, 32)
        self.rfb2 = RFB_modified(128, 32)
        self.rfb1 = RFB_modified(64, 32)
        # self.decoder1 = decoder1(32)
        # self.decoder2 = decoder2(384)
        self.transformerdecoder = transfmrerDecoder(6, 4, 32)
        self.qry = nn.Parameter(torch.zeros(12, 4, 32))
        self.bn1 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True)
        self.bn2 = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True)
        self.bn12 = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True)
        conv = nn.Sequential()
        conv.add_module('conv1', nn.Conv2d(384, 96, 3, 1, 1))
        conv.add_module('bn1', self.bn1)
        conv.add_module('relu1', nn.ReLU(inplace=True))
        conv.add_module('conv2', nn.Conv2d(96, 1, 3, 1, 1))
        conv.add_module('bn2', self.bn2)
        conv.add_module('relu2', nn.ReLU(inplace=True))
        self.conv = conv
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(384, 384, kernel_size=3, padding=1))
        self.conv_last = nn.Conv2d(384, 1, kernel_size=3, padding=1)
        self.upsample0 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.upsample1 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=4, mode='bilinear')


        #rgb
        self.rgb_encoder = pvt_v2_b2()
        self.rfb33 = RFB_modified(512, 32)
        self.rfb22 = RFB_modified(320, 32)
        self.rfb11 = RFB_modified(128, 32)
        self.rfb00 = RFB_modified(64, 32)
        self.rgs = nn.ModuleList()

        for i in range(4):
            self.rgs.append(Rgb_guide_sa())


        #fuse
        self.mhsa2 = Block(384, 6)
        self.mhsa3 = Block(384, 6)


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.weight.data.normal_(std=0.01)
        #         m.bias.data.fill_(0)
    def forward(self, x, y):

        #focal
        ba = x.size()[0]//12
        x = self.focal_encoder(x)
        x0f = self.rfb1(x[0])                        # [ba*12, 32, 64, 64]
        x1f = self.rfb2(x[1])                        # [ba*12, 32, 32, 32]
        x2f = self.rfb3(x[2])                        # [ba*12, 32, 16, 16]
        x3f = self.rfb4(x[3])                        # [ba*12, 32, 8, 8]
        # print(x0f.shape)
        x0f_re = x0f.reshape(12*ba, 32, 64*64).permute(0, 2, 1)
        x1f_re = x1f.reshape(12*ba, 32, 32*32).permute(0, 2, 1)
        x2f_re = x2f.reshape(12*ba, 32, 16*16).permute(0, 2, 1)
        x3f_re = x3f.reshape(12*ba, 32, 8*8).permute(0, 2, 1)

        # k_v = self.decoder1(x0f, x1f, x2f, x3f)  # [12*ba, 32, 8, 8]
        # k_v = k_v.reshape(12*ba, 32, -1).permute(0, 2, 1).contiguous()  # [12*ba, 64, 32]
        k_v = torch.cat((x0f_re, x1f_re), dim=1)
        k_v = torch.cat((k_v, x2f_re), dim=1)
        k_v = torch.cat((k_v, x3f_re), dim=1)
        # print(k_v.shape)  # [24, 5440, 32]

        qry = self.qry.repeat(ba, 1, 1)
        qry = self.transformerdecoder(qry, k_v)  # [24, 16, 32]
        qry = torch.softmax(torch.cat(torch.chunk(qry.unsqueeze(1), ba, dim=0), dim=1), dim=0)  # [12, 2, 16, 32]
        # qry = torch.cat(torch.chunk(qry.unsqueeze(1), ba, dim=0), dim=1)  # [12, 2, 16, 32]
        qry = qry.reshape(12, ba, 2, 2, -1).permute(0, 1, 4, 2, 3).contiguous()  # [12, 2, 32, 4, 4]
        qry = torch.cat(torch.chunk(qry, ba, dim=1), dim=0).squeeze(1)   # [12*ba, 32, 4, 4]

        qry0 = self.upsample0(qry)
        qry1 = self.upsample1(qry)
        qry2 = self.upsample2(qry)
        qry3 = self.upsample3(qry)


        x0q = torch.mul(x0f, qry0)
        x1q = torch.mul(x1f, qry1)
        x2q = torch.mul(x2f, qry2)
        x3q = torch.mul(x3f, qry3)

        # xf = []
        # xq = []
        # xf.append(x0f)
        # xf.append(x1f)
        # xf.append(x2f)
        # xf.append(x3f)
        # xq.append(x0q)
        # xq.append(x1q)
        # xq.append(x2q)
        # xq.append(x3q)

        # out_xf = x0f
        out_xq = x0q


        x0q_sal = torch.cat(torch.chunk(x0q.unsqueeze(1), ba, dim=0), dim=1)    # [12, ba, 32, 64, 64]
        x0q_sal = torch.cat(torch.chunk(x0q_sal, 12, dim=0), dim=2).squeeze(0)  # [ba, 384, 64, 64]
        x[0] = x0q_sal
        x0q_sal = self.conv(x0q_sal)                                            # [ba, 1, 64, 64]

        x1q_sal = torch.cat(torch.chunk(x1q.unsqueeze(1), ba, dim=0), dim=1)    # [12, ba, 32, 32, 32]
        x1q_sal = torch.cat(torch.chunk(x1q_sal, 12, dim=0), dim=2).squeeze(0)  # [ba, 384, 32, 32]
        x[1] = x1q_sal
        x1q_sal = self.conv(x1q_sal)                                            # [ba, 1, 32, 32]

        x2q_sal = torch.cat(torch.chunk(x2q.unsqueeze(1), ba, dim=0), dim=1)    # [12, ba, 32, 16, 16]
        x2q_sal = torch.cat(torch.chunk(x2q_sal, 12, dim=0), dim=2).squeeze(0)  # [ba, 384, 16, 16]
        x2_a = x2q_sal.reshape(ba, 384, 16*16).permute(0, 2, 1)
        x[2] = x2q_sal
        x2q_sal = self.conv(x2q_sal)                                            # [ba, 1, 16, 16]

        x3q_sal = torch.cat(torch.chunk(x3q.unsqueeze(1), ba, dim=0), dim=1)    # [12, ba, 32, 8, 8]
        x3q_sal = torch.cat(torch.chunk(x3q_sal, 12, dim=0), dim=2).squeeze(0)  # [ba, 384, 8, 8]
        x3_a = x3q_sal.reshape(ba, 384, 8 * 8).permute(0, 2, 1)
        x[3] = x3q_sal
        x3q_sal = self.conv(x3q_sal)                                            # [ba, 1, 8, 8]


        # out_focal = self.decoder2(x3q, x2q, x1q, x0q)                                 # [12*ba, 32, 64, 64]
        # out_focal = torch.cat(torch.chunk(out_focal.unsqueeze(1), ba, dim=0), dim=1)  # [12, ba, 32, 64, 64]
        # out_focal = torch.cat(torch.chunk(out_focal, 12, dim=0), dim=2).squeeze(0)    # [ba, 384, 64, 64]
        # out_focal = self.conv(out_focal)                                              # [ba, 1, 64, 64]
        # out_focal = F.interpolate(out_focal, size=(256, 256), mode='bilinear', align_corners=False)


        #rgb
        y = self.rgb_encoder(y)
        y[0] = self.rfb00(y[0])  # [ba, 32, 64, 64]
        y[1] = self.rfb11(y[1])  # [ba, 32, 32, 32]
        y[2] = self.rfb22(y[2])  # [ba, 32, 16, 16]
        y2_a = y[2].reshape(ba, 32, 16*16).permute(0, 2, 1)
        y[3] = self.rfb33(y[3])  # [ba, 32, 8, 8]
        y3_a = y[3].reshape(ba, 32, 8*8).permute(0, 2, 1)
        out_xf = y[0]
        xy2 = self.mhsa2(x2_a, y2_a)  # [2, 512, 384]
        xy2_fuse = xy2[:, 0:256, :] + xy2[:, 256:, :]  # [2, 256, 384]
        xy3 = self.mhsa3(x3_a, y3_a)  # [2, 128, 384]
        xy3_fuse = xy3[:, 0:64, :] + xy3[:, 64:, :]  # [2, 256, 384]


        xy2_fuse = xy2_fuse.reshape(ba, 16, 16, -1).permute(0, 3, 1, 2).contiguous()  # [2, 384, 16, 16]
        xy3_fuse = xy3_fuse.reshape(ba, 8, 8, -1).permute(0, 3, 1, 2).contiguous()  # [2, 384, 8, 8]

        xy3_fuse = F.interpolate(xy3_fuse, scale_factor=2, mode='bilinear', align_corners=False)

        xy23 = xy2_fuse + xy3_fuse

        s = []

        for i in range(1, -1, -1):
            if i == 1:
                r = self.rgs[i](y[i], x[i], xy23)
            else:
                r = self.rgs[i](y[i], x[i], s[0])
            s.insert(0, r)

        #print(s[3].shape, s[2].shape, s[1].shape, s[0].shape)  torch.Size([2, 384, 8, 8]) torch.Size([2, 384, 16, 16]) torch.Size([2, 384, 32, 32]) torch.Size([2, 384, 64, 64])

        # fuse_sal = self.decoder2(s[3], s[2], s[1], s[0])  # [2, 384, 64, 64]
        sde = s[0]
        fuse_sal = self.conv_last(s[0])  # [2, 1, 64, 64]
        fuse_pred = F.interpolate(fuse_sal, size=(256, 256), mode='bilinear', align_corners=False)  # [2, 1, 256, 256]


        return x0q_sal, x1q_sal, x2q_sal, x3q_sal, fuse_pred, out_xf, out_xq, sde, fuse_sal


# class model(nn.Module):
#     def __init__(self):
#         super(model, self).__init__()
#         self.focal_model = focal_model()

if __name__ == '__main__':
    import torchvision
    from ptflops import get_model_complexity_info
    import time

    from torchstat import stat
    # path = "../config/hrt_base.yaml"
    a = torch.randn(24, 3, 256, 256).cuda()
    b = torch.randn(2, 3, 256, 256).cuda()
    # c = torch.randn(1, 1, 352, 352).cuda()
    # config = yaml.load(open(path, "r"),yaml.SafeLoader)['MODEL']['HRT']
    # hr_pth_path = r"E:\ScientificResearch\pre_params\hrt_base.pth"
    # cnn_pth_path = r"D:\tanyacheng\Experiments\pre_trained_params\swin_base_patch4_window7_224_22k.pth"
    # cnn_pth_path = r"E:\ScientificResearch\pre_params\resnet18-5c106cde.pth"
    model = model().cuda()
    # model.initialize_weights()

    # out = model(a, b)

    # stat(model, (b, a))
    # 分析FLOPs
    # flops = FlopCountAnalysis(model, (a, b))
    # print("FLOPs: ", flops.total())
    #
    # # 分析parameters
    # print(parameter_count_table(model))

    # -- coding: utf-8 --


    # model = torchvision.pvt_v2_b2().alexnet(pretrained=False)
    # flops, params = get_model_complexity_info(model, a, as_strings=True, print_per_layer_stat=True)
    # print('flops: ', flops, 'params: ', params)

    # params, flops = profile(model, inputs=(b,))
    # params, flops = clever_format([params, flops], "%.2f")
    #
    # print(params, flops)
    # print(out.shape)
    # for x in out:
    #     print(x.shape)


    ###### FPS


    # nums = 710
    # time_s = time.time()
    # for i in range(nums):
    #     _ = model(a, b, c)
    # time_e = time.time()
    # fps = nums / (time_e - time_s)
    # print("FPS: %f" % fps)