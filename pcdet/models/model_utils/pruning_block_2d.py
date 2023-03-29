import re
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
#import MinkowskiEngine as ME
import spconv.pytorch as spconv

from functools import partial
from spconv.core import ConvAlgo
import copy
import time
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from pcdet.models.backbones_3d.focal_sparse_conv.focal_sparse_utils import FocalLoss

import cv2
import numpy as np

def sort_by_indices(features_foreground_cat, indices_foreground_coords, additional_features=None):
    a = indices_foreground_coords[:, 1:]
    # print("a shape:", a.shape)
    augmented_a = a.select(1, 0) * a[:, 1].max() + a.select(1, 1)
    augmented_a_sorted, ind = augmented_a.sort()
    features_foreground_cat = features_foreground_cat[ind]
    indices_foreground_coords = indices_foreground_coords[ind]
    if not additional_features is None:
        additional_features = additional_features[ind]
    return features_foreground_cat, indices_foreground_coords, additional_features

def check_repeat(x_foreground_features, x_foreground_indices, additional_features=None, sort_first=True, flip_first=True):
    if sort_first:
        x_foreground_features, x_foreground_indices, additional_features = sort_by_indices(x_foreground_features, x_foreground_indices, additional_features)

    if flip_first:
        x_foreground_features, x_foreground_indices = x_foreground_features.flip([0]), x_foreground_indices.flip([0])

    if not additional_features is None:
        additional_features=additional_features.flip([0])

    a = x_foreground_indices[:, 1:].int()
    augmented_a = torch.add(a.select(1, 0) * a[:, 1].max(), a.select(1, 1))
    _unique, inverse, counts = torch.unique_consecutive(augmented_a, return_inverse=True, return_counts=True, dim=0)
    
    if _unique.shape[0] < x_foreground_indices.shape[0]:
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        x_foreground_features_new = torch.zeros((_unique.shape[0], x_foreground_features.shape[-1]), device=x_foreground_features.device)
        x_foreground_features_new.index_add_(0, inverse.long(), x_foreground_features)
        x_foreground_features = x_foreground_features_new
        perm_ = inverse.new_empty(_unique.size(0)).scatter_(0, inverse, perm)
        x_foreground_indices = x_foreground_indices[perm_].int()

        if not additional_features is None:
            additional_features_new = torch.zeros((_unique.shape[0],), device=additional_features.device)
            additional_features_new.index_add(0, inverse.long(), additional_features)
            additional_features = additional_features_new / counts
    return x_foreground_features, x_foreground_indices, additional_features

def split_voxels_v2(x, b, voxel_importance, kernel_offsets, mask_multi=True, pruning_mode="topk", pruning_ratio=0.5):
    index = x.indices[:, 0]
    batch_index = index==b
    indices_ori = x.indices[batch_index]
    features_ori = x.features[batch_index]
    voxel_importance = voxel_importance[batch_index]

    if mask_multi:
        features_ori *= voxel_importance

    # get mask
    # print("pruning_mode-----------------------:", pruning_mode)
    if pruning_mode == "topk":
        _, indices = voxel_importance.view(-1,).sort()
        indices_im = indices[int(voxel_importance.shape[0]*pruning_ratio):]
        indices_nim = indices[:int(voxel_importance.shape[0]*pruning_ratio)]
        # print("indices_im num:", indices_im.sum(), "indices_nim num:",indices_nim.sum(), "pruning_ratio:", pruning_ratio, "x shape:", x.features.shape)
        # print("indices_im num:", indices_im.shape, "indices_nim num:",indices_nim.shape, "pruning_ratio:", pruning_ratio, "x shape:", x.features.shape)
    elif pruning_mode == "thre":
        indices_im = (voxel_importance.view(-1,) > pruning_ratio)
        indices_nim = (voxel_importance.view(-1,) <= pruning_ratio)
        # print("indices_im num:", indices_im.sum(), "indices_nim num:",indices_nim.sum(), "pruning_ratio:", pruning_ratio, "x shape:", x.features.shape)

    features_im = features_ori[indices_im]
    coords_im = indices_ori[indices_im]
    voxel_kerels_offset = kernel_offsets.unsqueeze(0).repeat(features_im.shape[0],1, 1) # [features_im.shape[0], 8, 2]
    indices_im_kernels = coords_im[:, 1:].unsqueeze(1).repeat(1, kernel_offsets.shape[0], 1) # [coords_im.shape[0], 8, 2]
    # print("kernel_offsets:", kernel_offsets.dtype, "indices_im_kernels:", indices_im_kernels.dtype, "voxel_kerels_offset:", voxel_kerels_offset.dtype)
    indices_with_imp = (indices_im_kernels + voxel_kerels_offset).view(-1, 2)

    spatial_indices = (indices_with_imp[:, 0] >0) * (indices_with_imp[:, 1] >0) * \
                        (indices_with_imp[:, 0] < x.spatial_shape[0]) * (indices_with_imp[:, 1] < x.spatial_shape[1])
    
    selected_indices = indices_with_imp[spatial_indices]
    selected_indices = torch.cat([torch.ones((selected_indices.shape[0], 1), device=features_im.device, dtype=torch.int)*b, selected_indices], dim=1)
    selected_features = torch.zeros((selected_indices.shape[0], features_ori.shape[1]), device=features_im.device)
    
    features_im = torch.cat([features_im, selected_features], dim=0) # [N', C]
    coords_im = torch.cat([coords_im, selected_indices], dim=0) # [N', 3]
    # mask_kernel_im = voxel_importance[indices_im][spatial_indices]
    # mask_kernel_im = mask_kernel_im.unsqueeze(1).repeat(1, kernel_offsets.shape[0], 1) 
    # mask_kernel_im = torch.cat([torch.ones(features_im_cat.shape[0], device=features_im.device), mask_kernel_im], dim=0)
    # print("before:", features_im.shape)
    assert features_im.shape[0] == coords_im.shape[0]
    if indices_im.sum()>0:
        features_im, coords_im, _ = check_repeat(features_im, coords_im)
        # print("after:", features_im.shape)
    # print("coords_im after:", coords_im.dtype)
    features_nim = features_ori[indices_nim]
    coords_nim = indices_ori[indices_nim]

    return features_im, coords_im, features_nim, coords_nim

class SparseSequentialBatchdict(spconv.SparseSequential):
    def __init__(self, *args, **kwargs):
        super(SparseSequentialBatchdict, self).__init__(*args, **kwargs)

    def forward(self, input, batch_dict=None):
        for k, module in self._modules.items():
            if module is None:
                continue
            input, batch_dict = module(input, batch_dict)
        return input, batch_dict


class DynamicFocalPruningConv(spconv.SparseModule):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 voxel_stride,
                 indice_key=None, 
                 stride=1, 
                 padding=0, 
                 bias=False, 
                 pruning_ratio=0.5,
                 pred_mode="attn_pred",
                 pred_kernel_size=None,
                 mask_caculate_mode="avg_pool",
                 loss_mode=None,
                 point_cloud_range=[-3, -40, 0, 1, 40, 70.4],
                 voxel_size = [0.1, 0.05, 0.05],
                 algo=ConvAlgo.Native,
                 pruning_mode="topk",
                 progress_subm_pruning=False):
        super().__init__()
        self.indice_key = indice_key
        self.pred_mode =  pred_mode
        self.pred_kernel_size = pred_kernel_size
        self.mask_caculate_mode = mask_caculate_mode
        self.progress_subm_pruning=progress_subm_pruning
        self.ori_pruning_ratio= pruning_ratio
        self.pruning_ratio = pruning_ratio
        self.kernel_size = kernel_size
        self.loss_mode = loss_mode
        self.inv_idx =  torch.Tensor([2, 1, 0]).long().cuda()
        
        self.pruning_mode = pruning_mode
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.voxel_stride = voxel_stride
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
    
        if pred_mode=="learnable":
            assert pred_kernel_size is not None
            self.pred_conv = spconv.SubMConv3d(
                    in_channels,
                    1,
                    kernel_size=pred_kernel_size,
                    stride=1,
                    padding=padding,
                    bias=False,
                    indice_key=indice_key + "_pred_conv",
                    algo=algo
                )

        self.conv_block = spconv.SubMConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        bias=bias,
                                        indice_key=indice_key,
                                        # subm_torch=False,
                                        algo=algo
                                    )
        if mask_caculate_mode == "avg_pool":
            self.avg_pool = spconv.SubMConv3d(
                                            out_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=0,
                                            bias=False,
                                            indice_key=indice_key + "_avg_pool",
                                            algo=algo
                                            # subm_torch=False,
                                        )
            new_weight = torch.ones_like(self.avg_pool.weight) / (kernel_size**3)
            channel_idx = torch.arange(out_channels).cuda()
            channel_mask = torch.eye(max(channel_idx) + 1)[channel_idx].view(1, 1, 1, out_channels, out_channels) # convert 2 onhot
            # print("new_weight shape:", new_weight.shape, "channel_mask shape:", channel_mask.shape)
            # assert False
            new_weight = new_weight * channel_mask
            self.avg_pool.weight = torch.nn.parameter.Parameter(new_weight)
            self.avg_pool.weight.requires_grad = False
        
        if self.loss_mode == "focal" or self.loss_mode == "focal_all":
            self.focal_loss = FocalLoss()
        self.sigmoid = nn.Sigmoid()

    def _combine_feature(self, x_im, x_nim, mask_position):
        # print("all :", mask_position.shape, "x shape:", x_im.features.shape, "masked:", mask_position.sum())
        # print("mask_position shape:", mask_position.shape, "x_im shape:", x_im.features.shape)
        assert x_im.features.shape[0] == x_nim.features.shape[0] == mask_position.shape[0]
        new_features = x_im.features
        new_features[mask_position] = x_nim.features[mask_position]
        x_im = x_im.replace_feature(new_features)
        return x_im
 
    def calulate_focal_loss(self, x, voxel_importance, batch_dict):
        spatial_indices = x.indices[:, 1:] * self.voxel_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]
        
        batch_size = x.batch_size
        mask_voxels = []
        box_of_pts_cls_targets = []
        
        for b in range(batch_size):
            if self.training:
                index=x.indices[:, 0]
                batch_index = index == b
                mask_voxel = voxel_importance[batch_index]
                voxels_3d_batch = voxels_3d[batch_index].unsqueeze(0)
                mask_voxels.append(mask_voxel)
                gt_boxes = batch_dict['gt_boxes'][b, :, :7].unsqueeze(0)
                box_of_pts_batch = points_in_boxes_gpu(voxels_3d_batch[:, :, self.inv_idx], gt_boxes).squeeze(0)
                box_of_pts_cls_targets.append(box_of_pts_batch>=0)
        
        loss_box_of_pts = 0
        if self.training:
            mask_voxels = torch.cat(mask_voxels).squeeze(-1)
            box_of_pts_cls_targets = torch.cat(box_of_pts_cls_targets)
            # print("box_of_pts_cls_targets:", box_of_pts_cls_targets.sum(), box_of_pts_cls_targets.shape)
            # print("mask_voxels", mask_voxels.shape, "box_of_pts_cls_targets:",box_of_pts_cls_targets.shape)
            mask_voxels_two_classes = torch.cat([1-mask_voxels.unsqueeze(-1), mask_voxels.unsqueeze(-1)], dim=1)
            # print("mask_voxels_two_classes", mask_voxels_two_classes.shape, "box_of_pts_cls_targets:",box_of_pts_cls_targets.shape)
            loss_box_of_pts = self.focal_loss(mask_voxels_two_classes, box_of_pts_cls_targets.long())

        return loss_box_of_pts

    def get_importance_mask(self, x, voxel_importance):
        batch_size = x.batch_size
        mask_position = torch.zeros(x.features.shape[0],).cuda()
        index = x.indices[:, 0]
        # print("self.pruning_ratio:", self.pruning_ratio)
        for b in range(batch_size):
            batch_index = index==b
            batch_voxel_importance = voxel_importance[batch_index]
            batch_mask_position = mask_position[batch_index]
            if self.pruning_mode == "topk":
                # print("check:------------")
                batch_mask_position_idx = torch.argsort(batch_voxel_importance.view(-1,))[:int(batch_voxel_importance.shape[0]*self.pruning_ratio)]
                batch_mask_position[batch_mask_position_idx] = 1
                # print("batch_mask_position_idx shape:", batch_mask_position_idx.shape, "batch_mask_position:", batch_mask_position_idx)
                mask_position[batch_index] =  batch_mask_position
                # print("check0:", mask_position.sum(), "mask value:", mask_position[0:10])
            elif self.pruning_mode == "thre":
                batch_mask_position_idx = (batch_voxel_importance.view(-1,) <= self.pruning_ratio)
                batch_mask_position[batch_mask_position_idx] = 1
                mask_position[batch_index] =  batch_mask_position
        # print("check1:", mask_position.sum(), "mask value:", mask_position[0:10])
        return mask_position.bool()

    def calculate_flops(self, x, batch_dict, mask_position):
        # mask_position = mask_position_ori
        if mask_position.dtype == torch.bool:
            mask_position = torch.nonzero(mask_position).view(-1,)
        pair_indices = copy.deepcopy(x.indice_dict[self.indice_key].indice_pairs)
        pair_indices_in = pair_indices[0] # [k**3, N]
        pair_indices_out = pair_indices[1] # [k**3, N]
        mask = torch.isin(pair_indices_out, mask_position)
        # print("before mask:", (pair_indices_out > -1).sum())
        pair_indices_out[mask] = -1
        # print("after mask:", (pair_indices_out > -1).sum())
        cur_flops = 2 * (pair_indices_out > -1).sum() * self.in_channels * self.out_channels - pair_indices_out.shape[1]
        batch_dict["3dbackbone_flops"] += cur_flops

    def write_obj(self, points, colors, out_filename):
        N = points.shape[0]
        fout = open(out_filename, 'w')
        for i in range(N):
            c = colors[i]
            fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
        fout.close()


    def visualize(self, x, x_attn_predict):
        new_x = copy.deepcopy(x)
        new_x = new_x.replace_feature(x_attn_predict)
        # map to dense
        dense_map = new_x.dense()
        N, C, D, H, W = dense_map.shape
        dense_map = dense_map.view(N, C * D, H, W).permute(0, 2, 3, 1)
        dense_map = torch.abs(dense_map).sum(-1) / (C * D)
        # min max norm
        min_v = dense_map.min()
        max_v = dense_map.max()
        norm_dense_map = (dense_map-min_v) / (max_v-min_v)
        print("max_v value:", max_v, "min_v value:", min_v, "norm_dense_map min:", norm_dense_map.min(), "norm_dense_map max:",norm_dense_map.max(), "norm_dense_map mean:", norm_dense_map.mean())
        norm_dense_map_np = norm_dense_map.squeeze(0).cpu().numpy()
        # print("norm_dense_map_np shape:", norm_dense_map_np.shape)
        map_vir = cm.get_cmap(name='viridis')
        color = map_vir(norm_dense_map_np)
        plt.imshow(color)
        plt.savefig("/data/home/jianhuiliu/newData/Research/CVMI-3D-Pruning/3d-detect-pruning-develop/output/visual/plot_" + self.indice_key+ "_" + str(x.features.shape[0]) + "_.png")

    def forward(self, x, batch_dict):
        # reset pruning ratio
        if self.progress_subm_pruning and self.training:
            cur_epoch=batch_dict["cur_epoch"]
            total_epochs=batch_dict["total_epochs"]
            self.pruning_ratio = (self.ori_pruning_ratio * (cur_epoch+1) / total_epochs)
            # print("pruning_ratio:", self.pruning_ratio)
        # pred importance
        if self.pred_mode=="learnable":
            x_ = x
            x_conv_predict = self.pred_conv(x_)
            voxel_importance = self.sigmoid(x_conv_predict.features) # [N, 1]
        elif self.pred_mode=="attn_pred":
            x_features = x.features
            x_attn_predict = torch.abs(x_features).sum(1) / x_features.shape[1]
            voxel_importance = self.sigmoid(x_attn_predict.view(-1, 1))
        else:
             raise Exception('pred_mode is not define')

        # self.visualize(x, voxel_importance)

        # get mask
        mask_position = self.get_importance_mask(x, voxel_importance)

        # conv
        x = x.replace_feature(x.features * voxel_importance)
        if self.mask_caculate_mode == "avg_pool":
            x_nim = self.avg_pool(x)
            x_im = self.conv_block(x)
        else:
            x_nim = x
            x_im = self.conv_block(x)
        
        # mask feature
        out = self._combine_feature(x_im, x_nim, mask_position)
        # caulate loss
        if self.training and (self.loss_mode == "focal" or self.loss_mode == "focal_all"):
            loss_box_of_pts = self.calulate_focal_loss(out, voxel_importance, batch_dict)
            batch_dict['loss_box_of_pts'] += loss_box_of_pts
        if self.training and self.loss_mode == "l1":
            # print("use_l1_loss-----------------")
            l1_loss = voxel_importance.mean()
            batch_dict['l1_loss'] += l1_loss
        # if not self.training:
        #     self.calculate_flops(out, batch_dict, mask_position)
        
        return out, batch_dict

class DynamicFocalPruningDownsample(spconv.SparseModule):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 indice_key=None, 
                 stride=1, 
                 padding=0, 
                 bias=False, 
                 pruning_ratio=0.5,
                 dilation=1,
                 voxel_stride=1,
                 point_cloud_range=[-3, -40, 0, 1, 40, 70.4],
                 voxel_size=[0.1, 0.05, 0.05],
                 pred_mode="attn_pred",
                 pred_kernel_size=None,
                 loss_mode=None,
                 algo=ConvAlgo.Native,
                 pruning_mode="topk",
                 progress_subm_downsample=False):
        super().__init__()
        if isinstance(padding, int):
            self.padding = [padding] * 3
        else:
            self.padding = padding
        self.indice_key = indice_key
        self.stride = stride
        self.dilation = dilation
        self.pred_mode =  pred_mode
        self.pred_kernel_size = pred_kernel_size
        # self.mask_caculate_mode = mask_caculate_mode
        self.progress_subm_downsample = progress_subm_downsample
        self.pruning_ratio = pruning_ratio
        self.origin_pruning_ratio = pruning_ratio
        self.kernel_size = kernel_size
        self.loss_mode = loss_mode
        self.inv_idx =  torch.Tensor([1, 0]).long().cuda()
        
        self.pruning_mode = pruning_mode
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.voxel_stride = voxel_stride
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()

        self.conv_block = spconv.SubMConv2d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=padding,
                                        bias=bias,
                                        indice_key=indice_key,
                                        # subm_torch=False,
                                        algo=algo
                                    )
        
        # self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        # self.voxel_size = torch.Tensor(voxel_size).cuda()
        _step = int(kernel_size//2)
        kernel_offsets = [[i, j] for i in range(-_step, _step+1) for j in range(-_step, _step+1)]
        kernel_offsets.remove([0, 0])
        self.kernel_offsets = torch.Tensor(kernel_offsets).cuda().int()    
        
        if self.loss_mode == "focal_sprs" or self.loss_mode == "focal_all":
            self.focal_loss = FocalLoss()

        self.sigmoid = nn.Sigmoid()
        
    def gemerate_sparse_tensor(self, x, voxel_importance):
        batch_size = x.batch_size
        voxel_features_im = []    
        voxel_indices_im = []
        voxel_features_nim = []    
        voxel_indices_nim = []
        # mask_kernel_list = []
        # print("self.kernel_offsets:", self.kernel_offsets, "dtype:", self.kernel_offsets.dtype)
        # print("batch_size:", batch_size)
        for b in range(batch_size):
            features_im, indices_im, features_nim, indices_nim = split_voxels_v2(x, b, voxel_importance, self.kernel_offsets, pruning_mode=self.pruning_mode, pruning_ratio=self.pruning_ratio)
            # print("voxel_importance:", voxel_importance.shape, "features_im shape:", features_im.shape, "indices_im shape:", indices_im.shape)
            # mask_kernel_list.append(mask_kernel)
            voxel_features_im.append(features_im)
            voxel_indices_im.append(indices_im)
            voxel_features_nim.append(features_nim)
            voxel_indices_nim.append(indices_nim)

        voxel_features_im = torch.cat(voxel_features_im, dim=0)
        voxel_indices_im = torch.cat(voxel_indices_im, dim=0)
        voxel_features_nim = torch.cat(voxel_features_nim, dim=0)
        voxel_indices_nim = torch.cat(voxel_indices_nim, dim=0)
        # mask_kernel = torch.cat(mask_kernel_list, dim=0)
        # print("voxel_features_im shape:", voxel_features_im.shape, "voxel_indices_im shape:", voxel_indices_im.shape, "indices.dtype:", voxel_indices_im.dtype)
        # print("voxel_indices_im:", voxel_indices_im[0:10])
        x_im = spconv.SparseConvTensor(voxel_features_im, voxel_indices_im, x.spatial_shape, x.batch_size)
        x_nim = spconv.SparseConvTensor(voxel_features_nim, voxel_indices_nim, x.spatial_shape, x.batch_size)
        
        return x_im, x_nim

    def combine_feature(self, x_im, x_nim, remove_repeat=True):
        x_features = torch.cat([x_im.features, x_nim.features], dim=0)
        x_indices = torch.cat([x_im.indices, x_nim.indices], dim=0)
        if remove_repeat:
            index = x_indices[:, 0]
            features_out_list = []
            indices_coords_out_list = []
            for b in range(x_im.batch_size):
                batch_index = index==b
                features_out, indices_coords_out, _ = check_repeat(x_features[batch_index], x_indices[batch_index], flip_first=False)
                # print("check before:", x_features[batch_index].shape, "check after:", features_out.shape)
                features_out_list.append(features_out)
                indices_coords_out_list.append(indices_coords_out)
            x_features = torch.cat(features_out_list, dim=0)
            x_indices = torch.cat(indices_coords_out_list, dim=0)
        
        x_im = x_im.replace_feature(x_features)
        x_im.indices = x_indices
        # print("x_im shape:", x_im.features.shape)
        return x_im

    def calulate_focal_loss(self, x, voxel_importance, batch_dict, voxel_stride):
        spatial_indices = x.indices[:, 1:] * voxel_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]
        # print("voxel_stride:", voxel_stride, "point_cloud_range:", self.point_cloud_range, "voxel_size:", self.voxel_size)
        batch_size = x.batch_size
        mask_voxels = []
        box_of_pts_cls_targets = []
        
        for b in range(batch_size):
            if self.training:
                index=x.indices[:, 0]
                batch_index = index == b
                mask_voxel = voxel_importance[batch_index]
                voxels_3d_batch = voxels_3d[batch_index].unsqueeze(0)
                mask_voxels.append(mask_voxel)
                gt_boxes = batch_dict['gt_boxes'][b, :, :7].unsqueeze(0)
                box_of_pts_batch = points_in_boxes_gpu(voxels_3d_batch[:, :, self.inv_idx], gt_boxes).squeeze(0)
                box_of_pts_cls_targets.append(box_of_pts_batch>=0)
        
        loss_box_of_pts = 0
        if self.training:
            mask_voxels = torch.cat(mask_voxels).squeeze(-1)
            box_of_pts_cls_targets = torch.cat(box_of_pts_cls_targets)
            # print("mask_voxels", mask_voxels.shape, "box_of_pts_cls_targets:",box_of_pts_cls_targets.sum())
            mask_voxels_two_classes = torch.cat([1-mask_voxels.unsqueeze(-1), mask_voxels.unsqueeze(-1)], dim=1)
            # print("mask_voxels_two_classes", mask_voxels_two_classes.shape, "box_of_pts_cls_targets:",box_of_pts_cls_targets.shape)
            loss_box_of_pts = self.focal_loss(mask_voxels_two_classes, box_of_pts_cls_targets.long())

        return loss_box_of_pts

    def reset_spatial_shape(self, x):
        indices = x.indices
        features = x.features
        conv_valid_mask = ((indices[:,1:] % 2).sum(1)==0)
        
        pre_spatial_shape = x.spatial_shape
        new_spatial_shape = []
        for i in range(len(x.spatial_shape)):
            size = (pre_spatial_shape[i] + 2 * self.padding[i] - self.dilation *
                    (self.kernel_size - 1) - 1) // self.stride + 1
            if self.kernel_size == -1:
                new_spatial_shape.append(1)
            else:
                new_spatial_shape.append(size)
        indices[:,1:] = indices[:,1:] // 2
        coords = indices[:,1:][conv_valid_mask]
        spatial_indices = (coords[:, 0] >0) * (coords[:, 1] >0) * \
            (coords[:, 0] < new_spatial_shape[0]) * (coords[:, 1] < new_spatial_shape[1])

        x = spconv.SparseConvTensor(features[conv_valid_mask][spatial_indices], indices[conv_valid_mask][spatial_indices].contiguous(), new_spatial_shape, x.batch_size)

        return x


    def forward(self, x, batch_dict):
        if self.progress_subm_downsample and self.training:
            cur_epoch = batch_dict["cur_epoch"]
            total_epochs = batch_dict["total_epochs"]
            self.pruning_ratio = (self.origin_pruning_ratio * (cur_epoch+1) / total_epochs)

        if self.pred_mode=="learnable":
            # print("check----------------------")
            x_ = x
            x_conv_predict = self.pred_conv(x_)
            voxel_importance = self.sigmoid(x_conv_predict.features) # [N, 1]
        elif self.pred_mode=="attn_pred":
            # print("pruning ratio:", self.pruning_ratio, "pruning_mode:", self.pruning_mode, "pred_mode:", self.pred_mode)
            x_features = x.features
            x_attn_predict = torch.abs(x_features).sum(1) / x_features.shape[1]
            voxel_importance = self.sigmoid(x_attn_predict.view(-1, 1))
        else:
             raise Exception('pred_mode is not define')

        x_im, x_nim = self.gemerate_sparse_tensor(x, voxel_importance)
        out = self.combine_feature(x_im, x_nim, remove_repeat=True)
        out = self.conv_block(out)

        out = self.reset_spatial_shape(out)
        # print("x shape:", x.features.shape, "x_im:", x_im.features.shape, "x_nim:", x_nim.features.shape, "out:", out.features.shape)
        if self.training and self.loss_mode == "l1":
            # print("use_l1_loss-----------------")
            l1_loss = voxel_importance.mean()
            batch_dict['l1_loss'] += l1_loss
        
        if self.training and (self.loss_mode == "focal_sprs" or self.loss_mode == "focal_all"):
            loss_box_of_pts = self.calulate_focal_loss(x, voxel_importance, batch_dict, self.voxel_stride//2)
            # print("loss_box_of_pts:", loss_box_of_pts)
            batch_dict['loss_box_of_pts_sprs'] += loss_box_of_pts
        
        return out, batch_dict
