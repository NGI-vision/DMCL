import torch
from torch import nn
import torch.nn.functional as F
# from torchsummary import summary
from .pvtv2 import pvt_v2_b4
import numpy as np
import cv2

# Guidance-based Gated Attention
class GGA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False):
        super(GGA, self).__init__()
        self.expected_channels = in_channels + out_channels  

        self.gate_conv = nn.Sequential(
            nn.BatchNorm2d(self.expected_channels), 
            nn.Conv2d(self.expected_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)

    def forward(self, in_feat, gate_feat):
        if in_feat.shape[1] != gate_feat.shape[1]:
            gate_feat = F.interpolate(gate_feat, size=in_feat.shape[2:], mode='bilinear', align_corners=True)
            gate_feat = nn.Conv2d(gate_feat.shape[1], in_feat.shape[1], kernel_size=1, bias=False).to(in_feat.device)(gate_feat)
        combined = torch.cat([in_feat, gate_feat], dim=1)  # [B, C1 + C2, H, W]
        if combined.shape[1] != self.expected_channels:
            self.gate_conv[0] = nn.BatchNorm2d(combined.shape[1]).to(combined.device)
        attention_map = self.gate_conv(combined)
        in_feat = (in_feat * (attention_map + 1))
        out_feat = self.out_conv(in_feat)
        return out_feat





# Convolutional Block Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channel, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelAttention(in_channel, reduction_ratio)
        self.SpatialGate = SpatialAttention()

    def forward(self, x):
        channel_att = self.ChannelGate(x)
        x = channel_att * x
        spatial_att = self.SpatialGate(x)
        x = spatial_att * x
        return x

# Multi-modal Feature Fusion
class MFF(nn.Module):
    def __init__(self, in_channel, reduction_ratio=16):
        super(MFF, self).__init__()
        self.fea_fus = CBAM(in_channel)

    def forward(self, img, depth):
        x = img + depth + (img * depth)
        x = self.fea_fus(x)
        return x



class feature_fuse(nn.Module):
    def __init__(self, in_channel=128, out_channel=128):
        super(feature_fuse, self).__init__()
        self.dim = in_channel
        self.out_dim = out_channel
        self.fuseconv = nn.Sequential(nn.Conv2d(2 * self.dim, self.out_dim, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(self.out_dim),
                                      nn.ReLU(True))
        self.conv = nn.Sequential(nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(self.out_dim),
                                  nn.ReLU(True))

    def forward(self, Ri, Di):
        assert Ri.ndim == 4
        RDi = torch.cat((Ri, Di), dim=1)
        RDi = self.fuseconv(RDi)
        RDi = self.conv(RDi)
        return RDi
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear', align_corners=True)
        out = img_features

        # edge_features = F.interpolate(edge, x_size[2:],
        #                               mode='bilinear', align_corners=True)
        # edge_features = self.edge_conv(edge_features)
        # out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class Edge_Module(nn.Module):
    def __init__(self, in_fea=[128, 320, 512], mid_fea=32):
        super(Edge_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # RGB 分支
        self.conv2_rgb = nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv4_rgb = nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv5_rgb = nn.Conv2d(in_fea[2], mid_fea, 1)

        # Depth 分支
        self.conv2_depth = nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv4_depth = nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv5_depth = nn.Conv2d(in_fea[2], mid_fea, 1)

        self.conv5_2 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_4 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_5 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)


        self.rcab = RCAB(mid_fea * 6)  # RGB 3 + Depth 3
        self.classifier = nn.Conv2d(mid_fea * 6, 1, kernel_size=3, padding=1)

    def forward(self, input, x2_rgb, x4_rgb, x5_rgb, x2_depth, x4_depth, x5_depth):
        _, _, h, w = input.size()

        # ---- RGB 分支 ----
        edge2_rgb = self.relu(self.conv5_2(self.relu(self.conv2_rgb(x2_rgb))))
        edge4_rgb = self.relu(self.conv5_4(self.relu(self.conv4_rgb(x4_rgb))))
        edge5_rgb = self.relu(self.conv5_5(self.relu(self.conv5_rgb(x5_rgb))))

        # ---- Depth 分支 ----
        edge2_depth = self.relu(self.conv5_2(self.relu(self.conv2_depth(x2_depth))))
        edge4_depth = self.relu(self.conv5_4(self.relu(self.conv4_depth(x4_depth))))
        edge5_depth = self.relu(self.conv5_5(self.relu(self.conv5_depth(x5_depth))))

        # ---- 上采样 ----
        edge2_rgb = F.interpolate(edge2_rgb, size=(h, w), mode='bilinear', align_corners=True)
        edge4_rgb = F.interpolate(edge4_rgb, size=(h, w), mode='bilinear', align_corners=True)
        edge5_rgb = F.interpolate(edge5_rgb, size=(h, w), mode='bilinear', align_corners=True)
        edge2_depth = F.interpolate(edge2_depth, size=(h, w), mode='bilinear', align_corners=True)
        edge4_depth = F.interpolate(edge4_depth, size=(h, w), mode='bilinear', align_corners=True)
        edge5_depth = F.interpolate(edge5_depth, size=(h, w), mode='bilinear', align_corners=True)

        # ---- 拼接 + 调制 + 输出 ----
        edge = torch.cat([edge2_rgb, edge4_rgb, edge5_rgb,
                          edge2_depth, edge4_depth, edge5_depth], dim=1)
        edge = self.rcab(edge)
        edge = self.classifier(edge)
        return edge

class Decoder(nn.Module):
    def __init__(self, dim=128):
        super(Decoder, self).__init__()
        self.dim = dim
        self.out_dim = dim
        self.fuse1 = MFF(in_channel=64)
        self.fuse2 = MFF(in_channel=128)
        self.fuse3 = MFF(in_channel=320)
        self.fuse4 = MFF(in_channel=320)
        
        self.gga_1 = GGA(64, 64)
        self.gga_2 = GGA(128, 128)
        self.gga_3 = GGA(320, 320)
        self.gga_4 = GGA(320, 320)


        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear")

        self.Conv43 = nn.Sequential(nn.Conv2d(640, 128, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(True),
                                    nn.Conv2d(128, 128, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(True))


        self.Conv432 = nn.Sequential(nn.Conv2d(2 * self.out_dim, self.out_dim, 1, 1, 0, bias=False),
                                     nn.BatchNorm2d(self.out_dim),
                                     nn.ReLU(True), nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(self.out_dim),
                                     nn.ReLU(True))
        self.Conv4321 = nn.Sequential(nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0, bias=False),  # 改为 192
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(True),
                                      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(True))


        self.sal_pred = nn.Sequential(nn.Conv2d(self.out_dim, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
                                      nn.ReLU(True),
                                      nn.Conv2d(64, 1, 3, 1, 1, bias=False))

        self.linear4 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.aspp_rgb = _AtrousSpatialPyramidPoolingModule(512, 320,
                                                       output_stride=16)
        self.aspp_depth = _AtrousSpatialPyramidPoolingModule(512, 320,
                                                       output_stride=16)
        self.after_aspp_conv_rgb = nn.Conv2d(320 * 5, 320, kernel_size=1, bias=False)
        self.after_aspp_conv_depth = nn.Conv2d(320 * 5, 320, kernel_size=1, bias=False)

        self.edge_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.rcab_sal_edge = RCAB(32 * 2)
        self.fused_edge_sal = nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False)
        self.sal_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(True)



    def forward(self, x,  feature_list, feature_list_depth, edges):
        R1, R2, R3, R4 = feature_list[0], feature_list[1], feature_list[2], feature_list[3]
        D1, D2, D3, D4 = feature_list_depth[0], feature_list_depth[1], feature_list_depth[2], feature_list_depth[3]

        R4 = self.aspp_rgb(R4)
        D4 = self.aspp_depth(D4)
        R4 = self.after_aspp_conv_rgb(R4)
        D4 = self.after_aspp_conv_depth(D4)

        RD1 = self.fuse1(R1, D1)
        RD2 = self.fuse2(R2, D2)
        RD3 = self.fuse3(R3, D3)
        RD4 = self.fuse4(R4, D4)
        

        RD1 = self.gga_1(RD1, RD1)  
        RD2 = self.gga_2(RD2, F.interpolate(RD1, size=RD2.shape[2:], mode="bilinear"))  
        RD3 = self.gga_3(RD3, F.interpolate(RD2, size=RD3.shape[2:], mode="bilinear"))
        RD4 = self.gga_4(RD4, F.interpolate(RD3, size=RD4.shape[2:], mode="bilinear"))

        
        RD43 = self.up2(RD4)
        RD3 = F.interpolate(RD3, size=RD43.shape[2:], mode="bilinear", align_corners=True)
        RD43 = torch.cat((RD43, RD3), dim=1)
        RD43 = self.Conv43(RD43)


        RD432 = self.up2(RD43)

        RD2 = F.interpolate(RD2, size=RD432.shape[2:], mode="bilinear", align_corners=True)
        RD432 = torch.cat((RD432, RD2), dim=1)
        RD432 = self.Conv432(RD432)

        RD4321 = self.up2(RD432)

        RD1 = F.interpolate(RD1, size=RD4321.shape[2:], mode="bilinear", align_corners=True)

        RD4321 = torch.cat((RD4321, RD1), dim=1)
        RD4321 = self.Conv4321(RD4321)  # [B, 128, 56, 56]

        sal_map = self.sal_pred(RD4321)
        sal_out = self.up4(sal_map)
        

        edge_out = self.edge_conv(edges)  
        edge_out = self.fused_edge_sal(edge_out)  
    
        reduce_weight = nn.Parameter(torch.randn(128, 320, 1, 1), requires_grad=True).cuda()
        RD4 = F.conv2d(RD4, weight=reduce_weight, bias=None, stride=1)  
        mask4 = F.interpolate(self.linear4(RD4), size=x.size()[2:], mode='bilinear', align_corners=False)
        mask3 = F.interpolate(self.linear3(RD43), size=x.size()[2:], mode='bilinear', align_corners=False)
        mask2 = F.interpolate(self.linear4(RD432), size=x.size()[2:], mode='bilinear', align_corners=False)


        return sal_out, mask4, mask3, mask2, edge_out


class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        key = []
        self.encoder_rgb = pvt_v2_b4()
        self.encoder_depth = pvt_v2_b4()
        self.decoder = Decoder(dim=128)
        self.edge_layer = Edge_Module()
        self.fuse_canny_edge = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.conv1_1 = nn.Conv2d(512, 320, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_3 = nn.Conv2d(320, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_5 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_7 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_9 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_11 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))

        self.conv2_1 = nn.Conv2d(512, 320, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_3 = nn.Conv2d(320, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_5 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_7 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_9 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_11 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input_rgb, input_depth, edges):

        rgb_feats = self.encoder_rgb(input_rgb)
        depth_feats = self.encoder_depth(input_depth)

        edge_pred = self.edge_layer(
            input_rgb,
            rgb_feats[1], rgb_feats[2], rgb_feats[3],
            depth_feats[1], depth_feats[2], depth_feats[3]
        ) 
        edges_resized = F.interpolate(edges, size=edge_pred.shape[2:], mode='bilinear', align_corners=True)
        edge_fused = torch.cat([edge_pred, edges_resized], dim=1)
        edge_fused = self.fuse_canny_edge(edge_fused)

        if self.training is True:
            sal1 = self.conv1_11(self.upsample(self.conv1_9(self.upsample(self.conv1_7(
                self.upsample(self.conv1_5(
                    self.upsample(self.conv1_3(
                        self.upsample(self.conv1_1(rgb_feats[3])))))))))))


            sal2 = self.conv2_11(self.upsample(self.conv2_9(self.upsample(self.conv2_7(
                self.upsample(self.conv2_5(
                    self.upsample(self.conv2_3(
                        self.upsample(self.conv2_1(depth_feats[3])))))))))))


            result_final, mask4, mask3, mask2, edge_out = self.decoder(input_rgb, rgb_feats, depth_feats, edge_fused)

            return result_final, mask4, mask3, mask2, torch.sigmoid(sal1), torch.sigmoid(sal2), edge_out, edge_pred, edges_resized
        else:
            result_final, mask4, mask3, mask2, edge_out = self.decoder(input_rgb, rgb_feats, depth_feats, edge_fused)
            return result_final, mask4, mask3, mask2, edge_out, edge_pred, edges_resized

