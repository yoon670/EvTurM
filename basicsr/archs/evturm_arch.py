import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .unet_arch import UNet
from .arch_util import flow_warp, ConvResidualBlocks, SmallUpdateBlock
from .spynet_arch import SpyNet
from basicsr.utils.nn_common import conv, upconv, resnet_block,ChannelPool,BasicConv
from basicsr.utils.DS import DS
import torch.utils.checkpoint as checkpoint
import os



@ARCH_REGISTRY.register()
class EvTurM(nn.Module):
    """EvTurM
       Note that: this class is for 4x VSR

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 30
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=30, spynet_path=None):
        super().__init__()

        # alignment
        self.spynet = SpyNet(spynet_path)



        self.num_feat = num_feat
        ch1 = num_feat
        ch2 = ch1 * 2
        ch3 = ch1 * 4
        ch4 = num_feat
        self.ch4 = ch4
        # propagation
        self.backward_trunk = ConvResidualBlocks(2*self.ch4, self.ch4, num_block)
        self.forward_trunk = ConvResidualBlocks(2*self.ch4, self.ch4, num_block)
        # feature extractor
        self.conv1_1 = conv(3, ch1, 3, stride=1)
        self.conv1_2 = conv(ch1, ch1, 3, stride=1)
        self.conv1_3 = conv(ch1, ch1, 3, stride=1)

        self.conv2_1 = conv(ch1, ch2, 3, stride=1)
        self.conv2_2 = conv(ch2, ch2, 3, stride=1)
        self.conv2_3 = conv(ch2, ch2, 3, stride=1)

        self.conv3_1 = conv(ch2, ch3, 3, stride=1)
        self.conv3_2 = conv(ch3, ch3, 3, stride=1)
        self.conv3_3 = conv(ch3, ch3, 3, stride=1)

        self.conv4_1 = conv(ch2, ch4, 3, stride=1)
        self.conv4_2 = conv(ch4, ch4, 3, stride=1)
        self.conv4_3 = conv(ch4, ch4, 3, stride=1)

        self.conv4_4 = nn.Sequential(
            conv(2 * ch4, ch4, 3),
            resnet_block(ch4, 3, res_num=3),
            conv(ch4, ch4, 3))

        self.kconv1_1 = conv(8, ch1, 3, stride=1)
        self.kconv1_2 = conv(ch1, ch1, 3, stride=1)
        self.kconv1_3 = conv(ch1, ch1, 3, stride=1)

        self.kconv2_1 = conv(ch1, ch2, 3, stride=1)
        self.kconv2_2 = conv(ch2, ch2, 3, stride=1)
        self.kconv2_3 = conv(ch2, ch2, 3, stride=1)

        self.kconv3_1 = conv(ch2, ch3, 3, stride=1)
        self.kconv3_2 = conv(ch3, ch3, 3, stride=1)
        self.kconv3_3 = conv(ch3, ch3, 3, stride=1)

        self.kconv4_1 = conv(ch2, ch4, 3, stride=1)
        self.kconv4_2 = conv(ch4, ch4, 3, stride=1)
        self.kconv4_3 = conv(ch4, ch4, 3, stride=1)

        # disparity map estimator
        self.DME = nn.Sequential(
            conv(ch4, ch4, 3),
            resnet_block(ch4, 3, res_num=3),
            conv(ch4, 1, kernel_size=3))
        
        # filter predictor
        self.conv_DME = conv(1, ch4, kernel_size=3)
        self.N = 3
        self.kernel_dim = self.N * (ch4 * 3 * 2) + self.N * ch4
        self.F = nn.Sequential(
            conv(ch4, ch4, kernel_size=3),
            resnet_block(ch4, kernel_size=3, res_num=3),
            conv(ch4, self.kernel_dim, kernel_size=3))
        
        self.bins=5
        self.e_conv = nn.Sequential(
            nn.Conv2d(self.bins, ch4, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch4),
            nn.ReLU()
        )
        self.trans = nn.Sequential(
            conv(ch4, ch4, 3), 
            nn.ReLU(), 
            conv(ch4, ch4, 3)
        )

        self.trans2 = nn.Sequential(
            conv(ch4, ch4, 3), 
            nn.ReLU()
        )
        self.F_mlp = nn.Sequential(nn.Linear(ch4, 2*ch4), nn.ReLU(), nn.Linear(2*ch4, ch4), nn.Sigmoid())
        ## Spatial Attention
        self.gate_rgb = nn.Conv2d(ch4, 1, kernel_size=1, bias=True)

        self.gernerate_kernel = nn.Sequential(
            nn.Conv2d(ch4, ch4, 3, 1, 1),
            nn.ReLU(), 
            nn.Conv2d(ch4, ch4*3**2, 1, 1)
        )
        # self.kconv = KernelConv2D.KernelConv2D(kernel_size=3)
        self.body_sa = nn.Sequential(
            conv(ch4, ch4, 3),
            nn.ReLU(),
            conv(ch4, ch4, 3)
        )

        ## compression
        self.compress = ChannelPool()
         ## event conv
        self.spatial_e = BasicConv(2, 1, 5, stride=1, padding=2, relu=False)
        # fusion operation
        self.conv1x1_fusion = nn.Conv2d(ch4*2, ch4, kernel_size=1)
        #reconstruction
        self.conv_hr = nn.Conv2d(ch4, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    
    def get_feat(self, x):
        b, n, c, h, w = x.size()
        feats_ = self.cnet(x.view(-1, c, h, w))
        h, w = feats_.shape[2:]
        feats_ = feats_.view(b, n, -1, h, w)

        return feats_
    def apply_dynamic_kernel(self,f_turb, kernel_event, kernel_size=3):
        B, C, H, W = f_turb.shape
        f_turb_unfold = F.unfold(f_turb, kernel_size=kernel_size, padding=kernel_size//2)  # (B, C*k*k, H*W)
        f_turb_weighted = f_turb_unfold * kernel_event.view(B, C * kernel_size * kernel_size, H * W)
        f_turb_new = F.fold(f_turb_weighted, output_size=(H, W), kernel_size=kernel_size, padding=kernel_size//2)
        return f_turb + f_turb_new

    def feature_fusion(self,f_event,f_turb):
        """
        进行特征融合，包括通道注意力、空间注意力、动态过滤和事件空间注意力等。
        
        参数:
        - f_event: 事件特征张量 (b, c, h, w)
        - f_turb: 湍流特征张量 (b, c, h, w)

        返回:
        - res: 融合后的特征张量 (b, c, h, w)
        """
        fused_feature = f_event + f_turb
        fused_feature = self.trans(fused_feature)
        ## - channel attention - ##
        w_0 = F.adaptive_avg_pool2d(fused_feature, (1,1)).squeeze()
        w_0 = self.F_mlp(w_0)
        w_0 = w_0.reshape(*w_0.shape, 1,1)
        f_turb = f_turb + w_0*f_turb
        ## - spatial attention - ##
        fused_feature2 = fused_feature + self.trans2(fused_feature)
        rgb_att = self.gate_rgb(fused_feature2)
        f_turb = f_turb + rgb_att*f_turb

        ### - dynamic filtering - ###
        kernel_event = self.gernerate_kernel(f_event)
        f_turb= self.apply_dynamic_kernel(f_turb, kernel_event)    
        ### - event spatial attention - ###
        sa_event = self.body_sa(f_event)
        e_compressed = self.compress(sa_event)
        e_out = self.spatial_e(e_compressed)
        scale_e = torch.sigmoid(e_out)
        f_event = f_event + scale_e*f_event
        ### - feature fusion - ###
        res = self.conv1x1_fusion(torch.cat((f_event, f_turb), dim=1))
        return res

    def forward(self, imgs, voxels_f):

        """Forward function of EvTurM

        Args:
            imgs: Input frames with shape (b, n, c, h, w). b is batch size. n is the number of frames, and c equals 3 (RGB channels).
            voxels_f: forward event voxel grids with shape (b, n-1, Bins, h, w). n-1 is intervals between n frames.
        """

        # flows_forward, flows_backward = self.get_flow(imgs)
        b, n, _, h, w = imgs.size()

        f_list=[]

        for i in range(0,n):
            bins = voxels_f.size()[2]
            C  = imgs[:,i,:,:,:]
            f_C = self.conv1_1(C)
            f_C = self.conv2_1(f_C)
            f_C = self.conv4_1(f_C)
            # 如果 i = 0，那么事件体素没有前一帧数据
            if i == 0:
                E = imgs.new_zeros(b, bins, h, w) # (b, 1, Bins, h, w)
            else:
                E = voxels_f[:, i-1, :, :, :]  # 选择第 i-1 帧的事件体素数据
            merged = torch.cat([C, E], dim=1)  # (b, 3 + Bins, h, w)

            f = self.kconv1_1(merged)
            f = self.kconv2_1(f)
            DM = self.DME(f)
            f_DM = self.conv_DME(DM)
            f = self.conv4_4(torch.cat([f, f_DM], 1))
            F = self.F(f)

            # Dynamic-Separable
            f = DS(f_C, F, self.N, self.ch4, 3)
            f_list.append(f)
        f_RGB = torch.stack(f_list, dim=1)
        flows_forward, flows_backward = self.get_flow(imgs)

        # backward branch
        out_l=[]
        feat_prop = f_RGB.new_zeros(b, self.ch4, h, w)
        for i in range( n - 1, -1 , -1 ):
            f_turb  = f_RGB[:,i,:,:,:]
            b,c, h, w = f_turb.size()
            if i == 0:
                f_event = f_turb.new_zeros(b, c, h, w) # (b, 1, Bins, h, w)
            else:
                f_event = self.e_conv(voxels_f[:, i-1, :, :, :])
            #DMCB
            res = self.feature_fusion(f_event, f_turb)

            if i< n - 1:
                flow = flows_backward[:, i, :, :, :] 
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                f_turb2 = flow_warp(f_RGB[:,i+1,:,:,:], flow.permute(0, 2, 3, 1))
                res2 = self.feature_fusion(f_event, f_turb2)
                feat_prop = torch.cat([res2, feat_prop], dim=1)
                feat_prop = self.backward_trunk(feat_prop)
            feat_prop = torch.cat([res, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)
        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            f_turb  = f_RGB[:,i,:,:,:]
            b,c, h, w = f_turb.size()
            if i == 0:
                f_event = f_turb.new_zeros(b, c, h, w) # (b, 1, Bins, h, w)
            else:
                f_event = self.e_conv(voxels_f[:, i-1, :, :, :])

            res = self.feature_fusion(f_event, f_turb)

            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                f_turb2 = flow_warp(f_RGB[:,i-1,:,:,:], flow.permute(0, 2, 3, 1))
                res2 = self.feature_fusion(f_event, f_turb2)
                feat_prop = torch.cat([res2, feat_prop], dim=1)
                feat_prop = self.forward_trunk(feat_prop)
            feat_prop = torch.cat([res, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)
            
            out = torch.cat([out_l[i], feat_prop], dim=1)
            out = self.forward_trunk(out)
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = imgs[:, i, :, :, :]
            out += base
            out_l[i] = out
        return torch.stack(out_l, dim=1)





            
            



