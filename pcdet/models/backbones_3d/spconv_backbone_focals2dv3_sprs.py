from functools import partial
import torch
import torch.nn as nn

from spconv.core import ConvAlgo
from ...utils.spconv_utils import replace_feature, spconv
#from .focal_sparse_conv.focal_sparse_conv_vote import FocalSparseConvVote
# from ...models.model_utils.pruning_block import DynamicPruningConv, DynamicPruningConvNoPool, DynamicPruningConvLinearPred, DynamicPruningConvAttnPred, DynamicFocalPruningConv
from ...models.model_utils.pruning_block import DynamicFocalPruningDownsample


class SparseSequentialBatchdict(spconv.SparseSequential):
    def __init__(self, *args, **kwargs):
        super(SparseSequentialBatchdict, self).__init__(*args, **kwargs)

    def forward(self, input, batch_dict=None):
        for k, module in self._modules.items():
            if module is None:
                continue
            input, batch_dict = module(input, batch_dict)
        return input, batch_dict

class PostActBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0, pruning_ratio=0.5,
                   conv_type='subm', norm_fn=None, loss_mode=None, algo=ConvAlgo.Native, downsample_pruning_mode="topk"):
        super().__init__()
        self.indice_key = indice_key
        self.in_channels = in_channels
        self.out_channels = out_channels
        # print("conv_type:", conv_type)
        # assert False
        if conv_type == 'subm':
            self.conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        elif conv_type == 'spconv':
            self.conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    bias=False, indice_key=indice_key)
        elif conv_type == 'inverseconv':
            self.conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        elif conv_type == "dynamicdownsample_attn":
            self.conv = DynamicFocalPruningDownsample(in_channels, out_channels, kernel_size, stride=stride, padding=padding, indice_key=indice_key, bias=False, 
                pruning_ratio=pruning_ratio, pred_mode="attn_pred", pred_kernel_size=None, loss_mode=loss_mode, algo=algo, pruning_mode=downsample_pruning_mode)
        else:
            raise NotImplementedError

        self.bn1 = norm_fn(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, batch_dict):
        if isinstance(self.conv, (DynamicFocalPruningDownsample,)):
            x, batch_dict = self.conv(x, batch_dict)
        else:
            x = self.conv(x)
        x = replace_feature(x, self.bn1(x.features))
        x = replace_feature(x, self.relu(x.features))
        return x, batch_dict


def post_act_block_dense(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_fn=None):
    m = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv2d(
            inplanes, planes, kernel_size=kernel_size, stride=stride, padding=int(kernel_size//2), bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv2d(
            planes, planes, kernel_size=kernel_size, stride=stride, padding=int(kernel_size//2), bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, batch_dict):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out, batch_dict


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None):
        super(BasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out
    

class PillarRes18BackBone8xFocals2DV3(nn.Module):
    downsample_type = ["spconv", "spconv", "spconv", "spconv", "spconv"]
    downsample_pruning_ratio = [0.5, 0.5, 0.5, 0.5, 0.5]
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]
        
        self.downsample_pruning_mode = model_cfg.get("DOWNSAMPLE_PRUNING_MODE", "topk")
        self.pruning_mode = model_cfg.get("PRUNING_MODE", "topk")

        block = PostActBlock
        dense_block = post_act_block_dense
        
        self.out_stage = model_cfg.get('OUT_STAGE', 4)

        kernel_sizes = model_cfg.get('KERNEL_SIZES', [3, 3, 3, 3, 3])
        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])

        self.conv1 = SparseSequentialBatchdict(
            SparseBasicBlock(32, 32, kernel_sizes[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(32, 32, kernel_sizes[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = SparseSequentialBatchdict(
            # [1600, 1408] <- [800, 704]
            block(32, 64, spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', conv_type=self.downsample_type[0], pruning_ratio=self.downsample_pruning_ratio[0]),
            SparseBasicBlock(64, 64, kernel_sizes[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, kernel_sizes[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = SparseSequentialBatchdict(
            # [800, 704] <- [400, 352]
            block(64, 128, spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv3', conv_type=self.downsample_type[1], pruning_ratio=self.downsample_pruning_ratio[1]),
            SparseBasicBlock(128, 128, kernel_sizes[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, kernel_sizes[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = SparseSequentialBatchdict(
            # [400, 352] <- [200, 176]
            block(128, 256, spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv4', conv_type=self.downsample_type[2], pruning_ratio=self.downsample_pruning_ratio[2]),
            SparseBasicBlock(256, 256, kernel_sizes[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, kernel_sizes[3], norm_fn=norm_fn, indice_key='res4'),
        )

        self.conv5 = SparseSequentialBatchdict(
            # [400, 352] <- [200, 176]
            block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv5', conv_type=self.downsample_type[3], pruning_ratio=self.downsample_pruning_ratio[3]),
            SparseBasicBlock(256, 256, kernel_sizes[4], norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(256, 256, kernel_sizes[4], norm_fn=norm_fn, indice_key='res5'),
        )

        self.conv6 = SparseSequentialBatchdict(
            # [400, 352] <- [200, 176]
            block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv6', conv_type=self.downsample_type[4], pruning_ratio=self.downsample_pruning_ratio[4]),
            SparseBasicBlock(256, 256, kernel_sizes[4], norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(256, 256, kernel_sizes[4], norm_fn=norm_fn, indice_key='res6'),
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(256, 256, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(256),
            nn.ReLU(),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }
        self.forward_ret_dict = {}

    def get_loss(self, tb_dict=None):
        loss = self.forward_ret_dict['loss_backbone']
        if tb_dict is None:
            tb_dict = {}
        tb_dict['loss_backbone'] = loss.item()
        return loss, tb_dict

    def combine_out(self, x_conv1, x_conv2, ratio=1):
        x_conv2.indices[:, 1:] *= ratio
        indices_cat = torch.cat([x_conv1.indices, x_conv2.indices])
        features_cat = torch.cat([x_conv1.features, x_conv2.features])

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=x_conv1.spatial_shape,
            batch_size=x_conv1.batch_size
        )
        return x_out

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        x_conv1, batch_dict = self.conv1(input_sp_tensor, batch_dict)
        x_conv2, batch_dict = self.conv2(x_conv1, batch_dict)
        x_conv3, batch_dict = self.conv3(x_conv2, batch_dict)
        x_conv4, batch_dict = self.conv4(x_conv3, batch_dict)
        x_conv5, batch_dict = self.conv5(x_conv4, batch_dict)
        x_conv6, batch_dict = self.conv6(x_conv5, batch_dict)

        if self.out_stage==4:
            out = self.combine_out(x_conv5, x_conv6, 2)
            out = self.combine_out(x_conv4, out, 2)
        elif self.out_stage==5:
            out = self.combine_out(x_conv5, x_conv6, 2)
        elif self.out_stage==6:
            out = x_conv6

        out = self.conv_out(out)
        self.forward_ret_dict['loss_backbone'] = torch.Tensor([0]).cuda()

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 2 ** (self.out_stage-1)
        })
        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })
        
        return batch_dict

class PillarRes18BackBone8xFocals2DV3SPRSRatio01(PillarRes18BackBone8xFocals2DV3):
    downsample_type = ["dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn"]
    downsample_pruning_ratio = [0.1, 0.1, 0.1, 0.1, 0.1]

class PillarRes18BackBone8xFocals2DV3SPRSRatio03(PillarRes18BackBone8xFocals2DV3):
    downsample_type = ["dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn"]
    downsample_pruning_ratio = [0.3, 0.3, 0.3, 0.3, 0.3]

class PillarRes18BackBone8xFocals2DV3SPRSRatio05(PillarRes18BackBone8xFocals2DV3):
    downsample_type = ["dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn"]
    downsample_pruning_ratio = [0.5, 0.5, 0.5, 0.5, 0.5]

class PillarRes18BackBone8xFocals2DV3SPRSRatio07(PillarRes18BackBone8xFocals2DV3):
    downsample_type = ["dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn"]
    downsample_pruning_ratio = [0.7, 0.7, 0.7, 0.7, 0.7]

class PillarRes18BackBone8xFocals2DV3SPRSRatio09(PillarRes18BackBone8xFocals2DV3):
    downsample_type = ["dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn", "dynamicdownsample_attn"]
    downsample_pruning_ratio = [0.9, 0.9, 0.9, 0.9, 0.9]
