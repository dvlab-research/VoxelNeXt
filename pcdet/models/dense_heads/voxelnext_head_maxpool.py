import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import centernet_utils
from ..model_utils import model_nms_utils
from ...utils import loss_utils
import spconv.pytorch as spconv
from spconv.core import ConvAlgo
import copy


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, kernel_size, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(spconv.SparseSequential(
                    spconv.SubMConv2d(input_channels, input_channels, kernel_size, padding=int(kernel_size//2), bias=use_bias, indice_key=cur_name),
                    nn.BatchNorm1d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(spconv.SubMConv2d(input_channels, output_channels, 1, bias=True, indice_key=cur_name+'out'))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, spconv.SubMConv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x, skip_key=[]):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            if cur_name in skip_key:
                continue
            ret_dict[cur_name] = self.__getattr__(cur_name)(x).features

        return ret_dict


class VoxelNeXtHeadMaxPool(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=False):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        self.gaussian_ratio = self.model_cfg.get('GAUSSIAN_RATIO', 1)
        self.gaussian_type = self.model_cfg.get('GAUSSIAN_TYPE', ['nearst', 'gt_center'])
        # The iou branch is only used for Waymo dataset
        self.iou_branch = self.model_cfg.get('IOU_BRANCH', False)
        if self.iou_branch:
            self.rectifier = self.model_cfg.get('RECTIFIER', [0.68, 0.71, 0.65])
            
        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        kernel_size_head = self.model_cfg.get('KERNEL_SIZE_HEAD', 3)

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.get('SHARED_CONV_CHANNEL', 128),
                    sep_head_dict=cur_head_dict,
                    kernel_size=kernel_size_head,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                )
            )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()
        self.voxel_size_list = self.model_cfg.VOXEL_SIZE_EACH_HEAD
        kernel_size_list = self.model_cfg.KERNEL_SIZE_EACH_HEAD
        self.max_pool_list = [spconv.SparseMaxPool2d(k, 1, 1, subm=True, algo=ConvAlgo.Native, indice_key='max_pool_head%d'%i) for i, k in enumerate(kernel_size_list)]
        self.easy_head = self.model_cfg.get('EASY_HEAD', [])

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossSparse())
        self.add_module('reg_loss_func', loss_utils.RegLossSparse())
        if self.iou_branch:
            self.add_module('crit_iou', loss_utils.IouLossSparse())
            self.add_module('crit_iou_reg', loss_utils.IouRegLossSparse())

    def assign_targets(self, gt_boxes, num_voxels, spatial_indices, spatial_shape):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:
        """
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': [],
            'gt_boxes': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, gt_boxes_list = [], [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head,
                    num_voxels=num_voxels[bs_idx], spatial_indices=spatial_indices[bs_idx], 
                    spatial_shape=spatial_shape, 
                    feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))
                gt_boxes_list.append(gt_boxes_single_head[:, :-1])

            ret_dict['heatmaps'].append(torch.cat(heatmap_list, dim=1).permute(1, 0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['gt_boxes'].append(gt_boxes_list)

        return ret_dict

    def distance(self, voxel_indices, center):
        distances = ((voxel_indices - center.unsqueeze(0))**2).sum(-1)
        return distances

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, num_voxels, spatial_indices, spatial_shape, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, num_voxels)

        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride

        coord_x = torch.clamp(coord_x, min=0, max=spatial_shape[1] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=spatial_shape[0] - 0.5)  #

        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= spatial_shape[1] and 0 <= center_int[k][1] <= spatial_shape[0]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            distance = self.distance(spatial_indices, center[k])
            inds[k] = distance.argmin()
            mask[k] = 1

            if 'gt_center' in self.gaussian_type:
                centernet_utils.draw_gaussian_to_heatmap_voxels(heatmap[cur_class_id], distance, radius[k].item() * self.gaussian_ratio)

            if 'nearst' in self.gaussian_type:
                centernet_utils.draw_gaussian_to_heatmap_voxels(heatmap[cur_class_id], self.distance(spatial_indices, spatial_indices[inds[k]]), radius[k].item() * self.gaussian_ratio)

            ret_boxes[k, 0:2] = center[k] - spatial_indices[inds[k]][:2]
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        batch_index = self.forward_ret_dict['batch_index']

        tb_dict = {}
        loss = 0
        batch_indices = self.forward_ret_dict['voxel_indices'][:, 0]
        spatial_indices = self.forward_ret_dict['voxel_indices'][:, 1:]

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes, batch_index
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()
            if self.iou_branch:
                batch_box_preds = self._get_predicted_boxes(pred_dict, spatial_indices)
                pred_boxes_for_iou = batch_box_preds.detach()
                iou_loss = self.crit_iou(pred_dict['iou'], target_dicts['masks'][idx], target_dicts['inds'][idx],
                                            pred_boxes_for_iou, target_dicts['gt_boxes'][idx], batch_indices)

                iou_reg_loss = self.crit_iou_reg(batch_box_preds, target_dicts['masks'][idx], target_dicts['inds'][idx],
                                                    target_dicts['gt_boxes'][idx], batch_indices)
                iou_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight'] if 'iou_weight' in self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS else self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                iou_reg_loss = iou_reg_loss * iou_weight #self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

                loss += (hm_loss + loc_loss + iou_loss + iou_reg_loss)
                tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()
                tb_dict['iou_reg_loss_head_%d' % idx] = iou_reg_loss.item()
            else:
                loss += hm_loss + loc_loss

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def _get_predicted_boxes(self, pred_dict, spatial_indices):
        center = pred_dict['center']
        center_z = pred_dict['center_z']
        #dim = pred_dict['dim'].exp()
        dim = torch.exp(torch.clamp(pred_dict['dim'], min=-5, max=5))
        rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
        rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
        angle = torch.atan2(rot_sin, rot_cos)
        xs = (spatial_indices[:, 1:2] + center[:, 0:1]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        ys = (spatial_indices[:, 0:1] + center[:, 1:2]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]

        box_part_list = [xs, ys, center_z, dim, angle]
        pred_box = torch.cat((box_part_list), dim=-1)
        return pred_box

    def _get_voxel_infos(self, x):
        spatial_shape = x.spatial_shape
        voxel_indices = x.indices
        spatial_indices = []
        num_voxels = []
        batch_size = x.batch_size
        batch_index = voxel_indices[:, 0]

        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            spatial_indices.append(voxel_indices[batch_inds][:, [2, 1]])
            num_voxels.append(batch_inds.sum())

        return spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels

    def _topk_1d(self, batch_size, batch_idx, obj, K=40):
        # scores: (N, num_classes)
        topk_score_list = []
        topk_inds_list = []
        topk_classes_list = []

        for bs_idx in range(batch_size):
            batch_inds = batch_idx==bs_idx
            score = obj[batch_inds].permute(1, 0)
            topk_scores, topk_inds = torch.topk(score, min(K, score.shape[-1]))
            topk_score, topk_ind = torch.topk(topk_scores.view(-1), min(K, topk_scores.view(-1).shape[-1]))

            topk_classes = (topk_ind // K).int()
            topk_inds = topk_inds.view(-1).gather(0, topk_ind)

            if not obj is None and obj.shape[-1] == 1:
                topk_score_list.append(obj[batch_inds][topk_inds])
            else:
                topk_score_list.append(topk_score)
            topk_inds_list.append(topk_inds)
            topk_classes_list.append(topk_classes)

        topk_score = torch.stack(topk_score_list)
        topk_inds = torch.stack(topk_inds_list)
        topk_classes = torch.stack(topk_classes_list)

        return topk_score, topk_inds, topk_classes

    def gather_feat(self, feats, inds, batch_size, batch_idx, batch_indice):
        feats_list = []
        dim = feats.size(-1)
        _inds = inds.unsqueeze(-1).expand(inds.size(0), dim)

        for bs_idx in range(batch_size):
            batch_inds = batch_idx==bs_idx
            feat = feats[batch_inds]
            bs_idx = batch_indice==bs_idx
            feats_list.append(feat.gather(0, _inds[bs_idx]))
        feats = torch.cat(feats_list)
        return feats

    def forward_test_waymo(self, x, data_dict):
        batch_index, spatial_indices = x.indices[:, 0], x.indices[:, 1:]

        K = self.model_cfg.POST_PROCESSING.MAX_OBJ_PER_SAMPLE
        score_thresh = self.model_cfg.POST_PROCESSING.SCORE_THRESH
        post_center_limit_range = torch.tensor(self.model_cfg.POST_PROCESSING.POST_CENTER_LIMIT_RANGE).cuda().float()
        batch_size = x.batch_size

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]

        for idx, head in enumerate(self.heads_list):
            scores_ori = head.hm(x).features.sigmoid()
            iou_ori = (head.iou(x).features + 1) * 0.5

            scores, inds, class_ids = self._topk_1d(x.batch_size, batch_index, scores_ori, K=K)
            mask = (scores > score_thresh).squeeze(-1)
            if mask.sum() == 0:
                continue
            scores = scores[mask].squeeze(-1)
            inds = inds[mask]
            class_ids = class_ids[mask]
            batch_indice = torch.arange(x.batch_size, device=x.indices.device).unsqueeze(-1).repeat(1, K)[mask].reshape(-1)
            spatial_idx = self.gather_feat(spatial_indices, inds, batch_size, batch_index, batch_indice)
            iou_preds = self.gather_feat(iou_ori, inds, batch_size, batch_index, batch_indice)
            skip_key = ['hm', 'iou', 'center']

            center = head.center(x).features
            center = self.gather_feat(center, inds, batch_size, batch_index, batch_indice) #.view(-1, 2) #.permute(2, 0, 1).view(2, -1).permute(1, 0)
            xs = (spatial_idx[:, -1:] + center[:, 0:1]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
            ys = (spatial_idx[:, -2:-1] + center[:, 1:2]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
            xys = torch.cat([xs, ys], dim=-1)

            selected = torch.zeros(scores.shape[0], device=scores.device)
            for cls in range(self.num_class):
                if mask_cls.sum() == 0:
                    continue
                mask_cls = class_ids == cls
                scores_cls = torch.pow(scores[mask_cls], 1 - self.rectifier[cls]) * torch.pow(iou_preds[mask_cls].squeeze(-1), self.rectifier[cls])
                scores[mask_cls] = scores_cls

                voxel_size = self.voxel_size_list[cls]
                xys_cls = xys[mask_cls]
                indices_cls = (xys_cls - xys_cls.min(0)[0]) / voxel_size
                bs_cls = batch_indice[mask_cls]
                indices_cls = torch.cat([bs_cls.unsqueeze(-1), indices_cls], dim=1).round().int()
                spatial_shape = indices_cls.max(0)[0][1:].int() + 1

                x_hm = spconv.SparseConvTensor(
                    features=scores_cls.unsqueeze(-1),
                    indices=indices_cls,
                    spatial_shape=spatial_shape,
                    batch_size=batch_size
                )
                max_pool = spconv.SparseMaxPool2d(self.kernel_size_list[cls], 1, 1, subm=True, algo=ConvAlgo.Native, indice_key='max_pool')
                x_hm_max = max_pool(x_hm, True)
                selected[mask_cls] += (x_hm_max.features == x_hm.features).squeeze(-1)

            selected = selected>=1
            pred_dict = head(x, skip_key=skip_key)
            center_z = pred_dict['center_z']
            dim = pred_dict['dim']
            rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)

            xs, ys = xs[selected], ys[selected]

            rot_sin = self.gather_feat(rot_sin, inds, batch_size, batch_index, batch_indice)[selected] #.view(-1, 1)[selected]
            rot_cos = self.gather_feat(rot_cos, inds, batch_size, batch_index, batch_indice)[selected] #.view(-1, 1)[selected]
            center_z = self.gather_feat(center_z, inds, batch_size, batch_index, batch_indice)[selected] #.view(-1, 1)[selected]
            dim = self.gather_feat(dim, inds, batch_size, batch_index, batch_indice)[selected].exp() #.view(-1, 3)[selected].exp()
            bs_indice = batch_indice[selected]

            angle = torch.atan2(rot_sin, rot_cos)

            box_part_list = [xs, ys, center_z, dim, angle]
            final_box_preds = torch.cat((box_part_list), dim=-1)
            final_class_ids = class_ids[selected]
            final_scores = scores[selected].squeeze(-1)

            mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(1)
            mask &= (final_scores > score_thresh)

            cur_boxes = final_box_preds[mask]
            cur_scores = final_scores[mask]
            cur_labels = final_class_ids[mask]
            cur_labels = self.class_id_mapping_each_head[idx][cur_labels.long()]
            bs_indice = bs_indice[mask]

            for k in range(batch_size):
                bs_idx = bs_indice==k
                _boxes, _scores, _labels = cur_boxes[bs_idx], cur_scores[bs_idx], cur_labels[bs_idx]

                ret_dict[k]['pred_boxes'].append(_boxes)
                ret_dict[k]['pred_scores'].append(_scores)
                ret_dict[k]['pred_labels'].append(_labels)

        for k in range(batch_size):
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        data_dict['final_box_dicts'] = ret_dict

        return data_dict

    def forward_test(self, x, data_dict):
        batch_index, spatial_indices = x.indices[:, 0], x.indices[:, 1:]

        K = self.model_cfg.POST_PROCESSING.MAX_OBJ_PER_SAMPLE
        score_thresh = self.model_cfg.POST_PROCESSING.SCORE_THRESH
        post_center_limit_range = torch.tensor(self.model_cfg.POST_PROCESSING.POST_CENTER_LIMIT_RANGE).cuda().float()
        batch_size = x.batch_size

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]

        for idx, head in enumerate(self.heads_list):
            scores_ori = head.hm(x).features.sigmoid()
            scores, inds, class_ids = self._topk_1d(x.batch_size, batch_index, scores_ori, K=K)
            mask = (scores > score_thresh).squeeze(-1)
            if mask.sum() == 0:
                continue
            scores = scores[mask].squeeze(-1)
            inds = inds[mask]
            class_ids = class_ids[mask]
            batch_indice = torch.arange(x.batch_size, device=x.indices.device).unsqueeze(-1).repeat(1, K)[mask].reshape(-1)
            spatial_idx = self.gather_feat(spatial_indices, inds, batch_size, batch_index, batch_indice)
            skip_key = ['hm']
            max_pool = self.max_pool_list[idx]

            if idx in self.easy_head:
                indices = self.gather_feat(x.indices, inds, batch_size, batch_index, batch_indice)
                indices[:, 0] = indices[:, 0] * 2 + class_ids
            else:
                voxel_size = self.voxel_size_list[idx]
                center = head.center(x).features
                center = self.gather_feat(center, inds, batch_size, batch_index, batch_indice) #.view(-1, 2) #.permute(2, 0, 1).view(2, -1).permute(1, 0)
                xs = (spatial_idx[:, -1:] + center[:, 0:1]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
                ys = (spatial_idx[:, -2:-1] + center[:, 1:2]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
                xys = torch.cat([xs, ys], dim=-1)

                indices = (xys - xys.min(0)[0]) / voxel_size
                bs_idx = batch_indice * 2 + class_ids
                indices = torch.cat([bs_idx.unsqueeze(-1), indices], dim=1).round().int()
                skip_key.append('center')

            spatial_shape = indices.max(0)[0][1:].int() + 1
            x_hm = spconv.SparseConvTensor(
                features=scores.unsqueeze(-1),
                indices=indices,
                spatial_shape=spatial_shape,
                batch_size=batch_size * 2
            )
            x_hm_max = max_pool(x_hm, True)
            selected = (x_hm_max.features == x_hm.features).squeeze(-1)

            pred_dict = head(x, skip_key=skip_key)
            center_z = pred_dict['center_z']
            dim = pred_dict['dim']
            rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            if not 'center' in skip_key:
                center = pred_dict['center']
                center = self.gather_feat(center, inds, batch_size, batch_index, batch_indice) #.view(-1, 2)
                xs = (spatial_idx[:, -1:] + center[:, 0:1]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
                ys = (spatial_idx[:, -2:-1] + center[:, 1:2]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]

            xs, ys = xs[selected], ys[selected]

            rot_sin = self.gather_feat(rot_sin, inds, batch_size, batch_index, batch_indice)[selected] #.view(-1, 1)[selected]
            rot_cos = self.gather_feat(rot_cos, inds, batch_size, batch_index, batch_indice)[selected] #.view(-1, 1)[selected]
            center_z = self.gather_feat(center_z, inds, batch_size, batch_index, batch_indice)[selected] #.view(-1, 1)[selected]
            dim = self.gather_feat(dim, inds, batch_size, batch_index, batch_indice)[selected].exp() #.view(-1, 3)[selected].exp()
            vel = self.gather_feat(vel, inds, batch_size, batch_index, batch_indice)[selected] #.view(-1, 2)[selected]
            bs_indice = batch_indice[selected]

            angle = torch.atan2(rot_sin, rot_cos)

            box_part_list = [xs, ys, center_z, dim, angle, vel]
            final_box_preds = torch.cat((box_part_list), dim=-1)
            final_class_ids = class_ids[selected]
            final_scores = scores[selected].squeeze(-1)

            mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(1)
            mask &= (final_scores > score_thresh)

            cur_boxes = final_box_preds[mask]
            cur_scores = final_scores[mask]
            cur_labels = final_class_ids[mask]
            cur_labels = self.class_id_mapping_each_head[idx][cur_labels.long()]
            bs_indice = bs_indice[mask]

            for k in range(batch_size):
                bs_idx = bs_indice==k
                _boxes, _scores, _labels = cur_boxes[bs_idx], cur_scores[bs_idx], cur_labels[bs_idx]

                ret_dict[k]['pred_boxes'].append(_boxes)
                ret_dict[k]['pred_scores'].append(_scores)
                ret_dict[k]['pred_labels'].append(_labels)

        for k in range(batch_size):
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

            if ret_dict[k]['pred_scores'].shape[0] > 500:
                scores, idx = ret_dict[k]['pred_scores'].sort(descending=True)
                idx = idx[:500]
                ret_dict[k]['pred_scores'] = ret_dict[k]['pred_scores'][idx]
                ret_dict[k]['pred_boxes'] = ret_dict[k]['pred_boxes'][idx]
                ret_dict[k]['pred_labels'] = ret_dict[k]['pred_labels'][idx]

        data_dict['final_box_dicts'] = ret_dict

        return data_dict

    def forward(self, data_dict):
        x = data_dict['encoded_spconv_tensor']

        if self.training:
            spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(x)            
            pred_dicts = []
            for head in self.heads_list:
                pred_dicts.append(head(x))
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], num_voxels, spatial_indices, spatial_shape
            )
            self.forward_ret_dict['batch_index'] = batch_index
            self.forward_ret_dict['target_dicts'] = target_dict
            self.forward_ret_dict['pred_dicts'] = pred_dicts
        else:
            forward_test = self.forward_test_waymo if self.iou_branch else self.forward_test
            data_dict = forward_test(x, data_dict)

        return data_dict
