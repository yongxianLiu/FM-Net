from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import math
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch
import torch.nn.functional as F
import ml_collections
from einops import rearrange
import numbers
from thop import profile
from model.SCTransNet.wtconv.wtconv2d import WTConv2d

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x)

class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.modulate = SFconv(inchannels)
    def forward(self, x):
        identity_input = x  # 3,32,64,64
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c // self.group,self.kernel_size ** 2, h * w)
        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)
        low_filter = self.act(low_filter)
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)
        out_high = identity_input - low_part
        out = self.modulate(low_part, out_high)
        return out
class SFconv(nn.Module):
    def __init__(self, features, M=2, r=2, L=32) -> None:
        super().__init__()
        d = max(int(features / r), L)
        self.features = features
        self.fc = nn.Conv2d(features, d, 1, 1, 0)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Conv2d(d, features, 1, 1, 0))
        self.softmax = nn.Softmax(dim=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Conv2d(features, features, 1, 1, 0)
    def forward(self, low, high):
        emerge = low + high
        emerge = self.gap(emerge)
        fea_z = self.fc(emerge)
        high_att = self.fcs[0](fea_z)
        low_att = self.fcs[1](fea_z)
        attention_vectors = torch.cat([high_att, low_att], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)
        fea_high = high * high_att
        fea_low = low * low_att
        out = self.out(fea_high + fea_low)
        return out
class Patch_ap(nn.Module):
    def __init__(self, inchannel, patch_size):
        super(Patch_ap, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.patch_size = patch_size
        self.channel = inchannel * patch_size ** 2
        self.h = nn.Parameter(torch.zeros(self.channel))
        self.l = nn.Parameter(torch.zeros(self.channel))
    def forward(self, x):
        patch_x = rearrange(x, 'b c (p1 w1) (p2 w2) -> b c p1 w1 p2 w2', p1=self.patch_size, p2=self.patch_size)
        patch_x = rearrange(patch_x, ' b c p1 w1 p2 w2 -> b (c p1 p2) w1 w2', p1=self.patch_size, p2=self.patch_size)
        low = self.ap(patch_x)
        high = (patch_x - low) * self.h[None, :, None, None]
        out = high + low * self.l[None, :, None, None]
        out = rearrange(out, 'b (c p1 p2) w1 w2 -> b c (p1 w1) (p2 w2)', p1=self.patch_size, p2=self.patch_size)
        return out

class Gap(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.fscale_d = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.fscale_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        x_d = self.gap(x)
        x_h = (x - x_d) * (self.fscale_h[None, :, None, None] + 1.)
        x_d = x_d  * self.fscale_d[None, :, None, None]
        return x_d + x_h
class ResBlock_FS(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock_FS, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.filter = filter
        self.dyna = dynamic_filter(in_channel // 2) if filter else nn.Identity()
        self.dyna_2 = dynamic_filter(in_channel // 2, kernel_size=5) if filter else nn.Identity() #kernel_size=1
        self.localap = Patch_ap(in_channel // 2, patch_size=2)
        self.global_ap = Gap(in_channel // 2)
    def forward(self, x):
        out = self.conv1(x)
        if self.filter:
            k3, k5 = torch.chunk(out, 2, dim=1)
            out_k3 = self.dyna(k3)
            out_k5 = self.dyna_2(k5)
            out = torch.cat((out_k3, out_k5), dim=1)
        non_local, local = torch.chunk(out, 2, dim=1)
        non_local = self.global_ap(non_local)
        local = self.localap(local)
        out = torch.cat((non_local, local), dim=1)
        out = self.conv2(out)
        return out + x

class Mask2FormerAttention(nn.Module):
    def __init__(self, channels):
        super(Mask2FormerAttention, self).__init__()
        self.channels = channels
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.mask = None
        self.norm = nn.LayerNorm([channels])

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        if channels != self.channels:
            raise ValueError("Input channel size does not match initialized channel size.")

        x = x.view(batch_size, channels, height * width).permute(0, 2, 1)

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.channels ** 0.5)

        if self.mask is None or self.mask.size(-1) != height * width or self.mask.size(0) != batch_size:
            binary_mask = torch.randint(0, 2, (batch_size, height, width), device=x.device)
            binary_mask = binary_mask.view(batch_size, -1)
            processed_mask = torch.where(binary_mask > 0.5, torch.tensor(0.0, device=x.device),
                                         torch.tensor(-float('inf'), device=x.device))
            self.mask = processed_mask.unsqueeze(1).expand(-1, height * width, -1)

        scores = scores + self.mask

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output + x
        attention_output = self.norm(attention_output)

        return attention_output.view(batch_size, channels, height, width)



class Channel_Embeddings(nn.Module):
    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 14 * 14 = 196

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        return x


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    # def forward(self, x, h, w):
    def forward(self, x):
        if x is None:
            return None

        x = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class ChannelTransformer(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[64, 128, 256, 512], patchSize=[32, 16, 8, 4]):
        super().__init__()

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]
        self.embeddings_1 = Channel_Embeddings(config, self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config, self.patchSize_2, img_size=img_size // 2, in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(config, self.patchSize_3, img_size=img_size // 4, in_channels=channel_num[2])
        self.embeddings_4 = Channel_Embeddings(config, self.patchSize_4, img_size=img_size // 8, in_channels=channel_num[3])

        self.maskAtt_encoder = Mask2FormerAttention(sum(channel_num))

        self.fuse1 = nn.Conv2d(sum(channel_num), channel_num[0], kernel_size=1)
        self.fuse2 = nn.Conv2d(sum(channel_num), channel_num[1], kernel_size=1)
        self.fuse3 = nn.Conv2d(sum(channel_num), channel_num[2], kernel_size=1)
        self.fuse4 = nn.Conv2d(sum(channel_num), channel_num[3], kernel_size=1)

        self.FSNet = ResBlock_FS(sum(channel_num),sum(channel_num),filter=True)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1, scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1, scale_factor=(self.patchSize_2, self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1, scale_factor=(self.patchSize_3, self.patchSize_3))
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1, scale_factor=(self.patchSize_4, self.patchSize_4))

    def forward(self, en1, en2, en3, en4):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        emb_all = torch.cat((emb1, emb2, emb3, emb4), dim=1)
        emb_all = self.maskAtt_encoder(emb_all)

        emb_all = self.FSNet(emb_all)

        encoded1 = self.fuse1(emb_all)
        encoded2 = self.fuse2(emb_all)
        encoded3 = self.fuse3(emb_all)
        encoded4 = self.fuse4(emb_all)

        x1 = self.reconstruct_1(encoded1) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None
        x3 = self.reconstruct_3(encoded3) if en3 is not None else None
        x4 = self.reconstruct_4(encoded4) if en4 is not None else None

        x1 = x1 + en1 if en1 is not None else None
        x2 = x2 + en2 if en2 is not None else None
        x3 = x3 + en3 if en3 is not None else None
        x4 = x4 + en4 if en4 is not None else None

        # return x1, x2, x3, x4, attn_weights
        return x1, x2, x3, x4

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)




class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv_middle = WTConv2d(out_channels, out_channels, kernel_size=3, stride=1, wt_levels=2, wt_type='haar'
                                                                                                              '')
        self.bn_middle = nn.BatchNorm2d(out_channels)
        # self.fca = FCA_Layer(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv_middle(out)
        out = self.bn_middle(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class FMNet(nn.Module):
    def __init__(self, config, n_channels=1, n_classes=1, img_size=256, vis=False, mode='train', deepsuper=True):
        super().__init__()
        self.vis = vis
        self.deepsuper = deepsuper
        print('Deep-Supervision:', deepsuper)
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel  # basic channel 64
        block = Res_block
        self.pool = nn.MaxPool2d(2, 2)
        self.inc = self._make_layer(block, n_channels, in_channels)
        self.down_encoder1 = self._make_layer(block, in_channels, in_channels * 2, 2)  # 64  128
        self.down_encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, 2)  # 64  128
        self.down_encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, 2)  # 64  128
        self.down_encoder4 = self._make_layer(block, in_channels * 8, in_channels * 8, 2)  # 64  128
        self.mtc = ChannelTransformer(config, vis, img_size,
                                      channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8],
                                      patchSize=config.patch_sizes)
        self.up_decoder4 = UpBlock_attention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up_decoder3 = UpBlock_attention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up_decoder2 = UpBlock_attention(in_channels * 4, in_channels, nb_Conv=2)
        self.up_decoder1 = UpBlock_attention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))

        if self.deepsuper:
            self.gt_conv5 = nn.Sequential(nn.Conv2d(in_channels * 8, 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(in_channels * 4, 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(in_channels * 1, 1, 1))
            self.outconv = nn.Conv2d(5 * 1, 1, 1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.inc(x)  # 64 224 224
        x2 = self.down_encoder1(self.pool(x1))  # 128 112 112
        x3 = self.down_encoder2(self.pool(x2))  # 256 56  56
        x4 = self.down_encoder3(self.pool(x3))  # 512 28  28
        d5 = self.down_encoder4(self.pool(x4))  # 512 14  14
        #  CCT
        f1 = x1
        f2 = x2
        f3 = x3
        f4 = x4
        #  CCT
        # x1, x2, x3, x4, att_weights = self.mtc(x1, x2, x3, x4)
        x1, x2, x3, x4 = self.mtc(x1, x2, x3, x4)
        x1 = x1 + f1
        x2 = x2 + f2
        x3 = x3 + f3
        x4 = x4 + f4
        #  Feature fusion
        d4 = self.up_decoder4(d5, x4)
        d3 = self.up_decoder3(d4, x3)
        d2 = self.up_decoder2(d3, x2)
        out = self.outc(self.up_decoder1(d2, x1))
        # deep supervision
        if self.deepsuper:
            gt_5 = self.gt_conv5(d5)
            gt_4 = self.gt_conv4(d4)
            gt_3 = self.gt_conv3(d3)
            gt_2 = self.gt_conv2(d2)
            # 原始深监督
            gt5 = F.interpolate(gt_5, scale_factor=16, mode='bilinear', align_corners=True)
            gt4 = F.interpolate(gt_4, scale_factor=8, mode='bilinear', align_corners=True)
            gt3 = F.interpolate(gt_3, scale_factor=4, mode='bilinear', align_corners=True)
            gt2 = F.interpolate(gt_2, scale_factor=2, mode='bilinear', align_corners=True)
            d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), 1))

            if self.mode == 'train':
                return (torch.sigmoid(gt5), torch.sigmoid(gt4), torch.sigmoid(gt3), torch.sigmoid(gt2), torch.sigmoid(d0), torch.sigmoid(out))
            else:
                return torch.sigmoid(out)
        else:
            return torch.sigmoid(out)
