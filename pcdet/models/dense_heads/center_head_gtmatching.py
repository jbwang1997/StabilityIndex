import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
from ...ops.roi_align_rotated import ROIAlignRotated
from functools import partial


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False, norm_func=None):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels) if norm_func is None else norm_func(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class CenterGTMatchingHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        norm_func = partial(nn.BatchNorm2d, eps=self.model_cfg.get('BN_EPS', 1e-5), momentum=self.model_cfg.get('BN_MOM', 0.1))
        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            norm_func(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                    norm_func=norm_func
                )
            )

        GT_MATCHING_CFG = self.model_cfg.get('GT_MATCHING_CFG', None)
        if GT_MATCHING_CFG is not None:
            self.roi_align = ROIAlignRotated(
                GT_MATCHING_CFG.ROI_SIZE, 1 / self.feature_map_stride, GT_MATCHING_CFG.SAMPLING_RATIO)
            self.gt_matching_loss = nn.MSELoss(reduction='mean')
            self.gt_matching_loss_weight = GT_MATCHING_CFG.LOSS_WEIGHT
        
        self.si_l1loss = nn.L1Loss(reduction='mean')
        self.si_l2loss = nn.MSELoss(reduction='mean')

        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()
        ret_boxes_src = gt_boxes.new_zeros(num_max_objs, gt_boxes.shape[-1])
        ret_boxes_src[:gt_boxes.shape[0]] = gt_boxes

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
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

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask, ret_boxes_src

    def assign_targets(self, gt_boxes, feature_map_size=None, gt_ids=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': [],
            'target_boxes_src': [],
            'gt_ids': [],
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, target_boxes_src_list = [], [], [], [], []
            gt_ids_list = []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                cur_gt_ids = gt_ids[bs_idx] if gt_ids is not None else None
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []
                gt_ids_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])
                    if gt_ids is not None:
                        gt_ids_single_head.append(cur_gt_ids[idx])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask, ret_boxes_src = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))
                target_boxes_src_list.append(ret_boxes_src.to(gt_boxes_single_head.device))
                if gt_ids is not None:
                    gt_ids_list.append(gt_ids_single_head)

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['target_boxes_src'].append(torch.stack(target_boxes_src_list, dim=0))
            ret_dict['gt_ids'].append(gt_ids_list)
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    @staticmethod
    def trans_box_by_matrix(box, matrix=None):
        """ Transform the box back to the original space
           @ box: (B, 7) [x, y, z, dx, dy, dz, rot]
           @ trans_matrix: (4, 4)

           rot is the angle between the box and the x axis
        """
        # if trans_matrix is None: return box
        # matrix = torch.inverse(trans_matrix)
        if matrix is None: return box

        # translate box center
        center = box[:, :3]
        center = torch.cat([center, center.new_ones(center.shape[0], 1)], dim=1)
        center = torch.matmul(center, matrix.t())
        recovered_center = center[:, :3]

        # translate the box size
        scale = torch.norm(matrix[0, :3], dim=0)
        recovered_dim = box[:, 3:6] * scale

        # rotate the box
        rot = box[:, 6:]
        fake_x, fake_y = torch.cos(rot), torch.sin(rot)
        fake_points = torch.cat([fake_x, fake_y, rot.new_ones(rot.shape[0], 2)], dim=1)
        # a dummy point of [0, 0, 1, 1]
        pivot = torch.cat([rot.new_zeros(1, 2), rot.new_ones(1, 2)], dim=1)
        pivot = torch.matmul(pivot, matrix.t())
        fake_points = torch.matmul(fake_points, matrix.t())
        fake_x_t, fake_y_t = fake_points[:, 0] - pivot[:, 0], fake_points[:, 1] - pivot[:, 1]
        recovered_rot = torch.atan2(fake_x_t, fake_y_t)

        recovered_box = torch.cat([recovered_center, recovered_dim, recovered_rot[:, None]], dim=1)
        return recovered_box
    
    def trans_box_by_matrix_batch(self, boxes, trans_matrix):
        batch_size = boxes.shape[0]
        output = []
        for i in range(batch_size):
            output.append(self.trans_box_by_matrix(boxes[i], trans_matrix[i]))
        return torch.stack(output, dim=0)

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0.0
        loss_feat_matching = 0.0

        batch_size = pred_dicts[0]['dim'].size(0)
        lidar_aug_matrix = target_dicts['lidar_aug_matrix']

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

            if 'iou' in pred_dict or self.model_cfg.get('IOU_REG_LOSS', False):
                batch_box_preds = centernet_utils.decode_bbox_from_pred_dicts(
                    pred_dict=pred_dict,
                    point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                    feature_map_stride=self.feature_map_stride
                )  # (B, H, W, 7 or 9)

                if 'iou' in pred_dict:
                    batch_box_preds_for_iou = batch_box_preds.permute(0, 3, 1, 2)  # (B, 7 or 9, H, W)

                    iou_loss = loss_utils.calculate_iou_loss_centerhead(
                        iou_preds=pred_dict['iou'],
                        batch_box_preds=batch_box_preds_for_iou.clone().detach(),
                        mask=target_dicts['masks'][idx],
                        ind=target_dicts['inds'][idx], gt_boxes=target_dicts['target_boxes_src'][idx]
                    )
                    loss += iou_loss
                    tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()

                if self.model_cfg.get('IOU_REG_LOSS', False):
                    iou_reg_loss = loss_utils.calculate_iou_reg_loss_centerhead(
                        batch_box_preds=batch_box_preds_for_iou,
                        mask=target_dicts['masks'][idx],
                        ind=target_dicts['inds'][idx], gt_boxes=target_dicts['target_boxes_src'][idx]
                    )
                    if target_dicts['masks'][idx].sum().item() != 0:
                        iou_reg_loss = iou_reg_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                        loss += iou_reg_loss
                        tb_dict['iou_reg_loss_head_%d' % idx] = iou_reg_loss.item()
                    else:
                        loss += (batch_box_preds_for_iou * 0.).sum()
                        tb_dict['iou_reg_loss_head_%d' % idx] = (batch_box_preds_for_iou * 0.).sum()

        # get the matching loss, update to tb_dict and sum to loss
        alignment = self.model_cfg.get('ALIGNMENT', None)
        if alignment is not None and alignment.get('ENABLE', False):
            loss_weight = alignment.get('LOSS_WEIGHT', [0, 0, 0, 0])
            assert len(loss_weight) == 4, 'loss_weight should be a list with length 4'

            for idx, pred_dict in enumerate(pred_dicts):
                ind = target_dicts['inds'][idx] # indexes to gather infos

                # heatmaps
                # make sure pred_dict['hm'] is already processed by sigmoid
                assert pred_dict['hm'].max() <= 1 and pred_dict['hm'].min() >= 0
                pred_hm, target_hm = pred_dict['hm'], target_dicts['heatmaps'][idx]
                pred_hm = pred_hm.gather(1, target_hm.argmax(dim=1).unsqueeze(1)).squeeze(1)
                pred_hm = pred_hm.reshape(batch_size, -1)
                score_preds = pred_hm.gather(1, ind)

                # boxes
                target_boxes_src = target_dicts['target_boxes_src'][idx][..., :-1]
                batch_box_preds = centernet_utils.decode_bbox_from_pred_dicts(
                    pred_dict=pred_dict,
                    point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                    feature_map_stride=self.feature_map_stride
                )  # (B, H, W, 7 or 9)
                box_preds = batch_box_preds.reshape(batch_size, -1, batch_box_preds.shape[-1])
                box_preds = box_preds.gather(1, ind.unsqueeze(2).expand(ind.size(0), ind.size(1), box_preds.size(2)))

                inverse_lidar_aug_matrix = torch.inverse(lidar_aug_matrix)
                recovered_target_boxes = self.trans_box_by_matrix_batch(target_boxes_src, inverse_lidar_aug_matrix)
                recovered_box_preds = self.trans_box_by_matrix_batch(box_preds, inverse_lidar_aug_matrix)

                ### (irving) remember to check this when using a new aug / a new dataset
                ### currently, the rotation angle is between the box and the x axis
                # check_err = (recovered_target_boxes[0, :10] - recovered_target_boxes[1, :10]).max()
                # if check_err > 1e-3:
                #     raise ValueError('The recovered target boxes are not the same')

                si_cls_loss, si_reg_loss = 0.0, 0.0
                gt_ids = target_dicts['gt_ids'][idx]
                for i in range(0, batch_size, 2):
                    gt_ids1, gt_ids2 = np.array(gt_ids[i]), np.array(gt_ids[i+1])
                    paired_idx1, paired_idx2 = np.nonzero(gt_ids1[:, None] == gt_ids2)
                    paired_idx1 = [idx for idx in paired_idx1 if len(gt_ids1[idx]) > 0]
                    paired_idx2 = [idx for idx in paired_idx2 if len(gt_ids2[idx]) > 0]

                    assert len(paired_idx1) == len(paired_idx2), 'The number of paired boxes should be the same'
                    if len(paired_idx1) == 0:
                        continue

                    # score losses
                    score_preds1, score_preds2 = score_preds[i][paired_idx1], score_preds[i+1][paired_idx2]
                    si_cls_loss += self.si_l2loss(score_preds1, score_preds2) * loss_weight[0]
                    
                    # reg losses
                    box_preds1, box_preds2 = recovered_box_preds[i][paired_idx1], recovered_box_preds[i+1][paired_idx2]
                    box_gts1, box_gts2 = recovered_target_boxes[i][paired_idx1], recovered_target_boxes[i+1][paired_idx2]

                    diff_loc1, diff_loc2 = box_preds1[:, :3] - box_gts1[:, :3], box_preds2[:, :3] - box_gts2[:, :3]
                    diff_dim1, diff_dim2 = box_preds1[:, 3:6] / box_gts1[:, 3:6], box_preds2[:, 3:6] / box_gts2[:, 3:6]
                    diff_rot1 = torch.stack([torch.sin(box_preds1[:, 6]) - torch.sin(box_gts1[:, 6]), 
                                             torch.cos(box_preds1[:, 6]) - torch.cos(box_gts1[:, 6])])
                    diff_rot2 = torch.stack([torch.sin(box_preds2[:, 6]) - torch.sin(box_gts2[:, 6]), 
                                             torch.cos(box_preds2[:, 6]) - torch.cos(box_gts2[:, 6])])

                    diff_loc1 = torch.stack([diff_loc1[:, 0] * torch.cos(box_gts1[:, 6]) + diff_loc1[:, 1] * torch.sin(box_gts1[:, 6]),
                                            -diff_loc1[:, 0] * torch.sin(box_gts1[:, 6]) + diff_loc1[:, 1] * torch.cos(box_gts1[:, 6]),
                                            diff_loc1[:, 2]], dim=1)

                    diff_loc2 = torch.stack([diff_loc2[:, 0] * torch.cos(box_gts2[:, 6]) + diff_loc2[:, 1] * torch.sin(box_gts2[:, 6]),
                                            -diff_loc2[:, 0] * torch.sin(box_gts2[:, 6]) + diff_loc2[:, 1] * torch.cos(box_gts2[:, 6]),
                                            diff_loc2[:, 2]], dim=1)

                    si_reg_loss += self.si_l1loss(diff_loc1, diff_loc2) * loss_weight[1]
                    si_reg_loss += self.si_l1loss(diff_dim1, diff_dim2) * loss_weight[2]
                    si_reg_loss += self.si_l1loss(diff_rot1, diff_rot2) * loss_weight[3]

                loss += si_cls_loss + si_reg_loss
                tb_dict['SI_cls_loss_head_%d' % idx] = si_cls_loss.item() if isinstance(si_cls_loss, torch.Tensor) else si_cls_loss
                tb_dict['SI_reg_loss_head_%d' % idx] = si_reg_loss.item() if isinstance(si_reg_loss, torch.Tensor) else si_reg_loss

        # feature matching loss
        if self.model_cfg.get('GT_MATCHING_CFG', None) is not None and \
            self.model_cfg.GT_MATCHING_CFG.get('ENABLE', False):
            gt_features = self.forward_ret_dict['gt_matching_dict']['gt_features']
            gt_ids = self.forward_ret_dict['gt_matching_dict']['gt_ids']
            batch_ids = self.forward_ret_dict['gt_matching_dict']['batch_ids']

            if gt_features.shape[0] != 0:
                paired_feature1 = []
                paired_feature2 = []
                for i in range(0, batch_size, 2):
                    features1 = gt_features[batch_ids == i]
                    features2 = gt_features[batch_ids == i+1]
                    gt_ids1 = gt_ids[(batch_ids == i).cpu().numpy()]
                    gt_ids2 = gt_ids[(batch_ids == i+1).cpu().numpy()]
                    paired_idx1, paired_idx2 = np.nonzero(gt_ids1[:, None] == gt_ids2)
                    paired_feature1.append(features1[paired_idx1])
                    paired_feature2.append(features2[paired_idx2])
                paired_feature1 = torch.cat(paired_feature1, dim=0)
                paired_feature2 = torch.cat(paired_feature2, dim=0)
                if paired_feature1.shape[0] != 0:
                    loss_feat_matching += self.gt_matching_loss(
                        paired_feature1, paired_feature2) * self.gt_matching_loss_weight

        # if loss_feat_matching is tensor
        tb_dict['gt_matching_loss'] = loss_feat_matching.item() if isinstance(loss_feat_matching, torch.Tensor) else loss_feat_matching
        loss += loss_feat_matching 

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            batch_iou = (pred_dict['iou'] + 1) * 0.5 if 'iou' in pred_dict else None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel, iou=batch_iou,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]

                if post_process_cfg.get('USE_IOU_TO_RECTIFY_SCORE', False) and 'pred_iou' in final_dict:
                    pred_iou = torch.clamp(final_dict['pred_iou'], min=0, max=1.0)
                    IOU_RECTIFIER = final_dict['pred_scores'].new_tensor(post_process_cfg.IOU_RECTIFIER)
                    final_dict['pred_scores'] = torch.pow(final_dict['pred_scores'], 1 - IOU_RECTIFIER[final_dict['pred_labels']]) * torch.pow(pred_iou, IOU_RECTIFIER[final_dict['pred_labels']])

                if post_process_cfg.NMS_CONFIG.NMS_TYPE not in  ['circle_nms', 'class_specific_nms']:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'class_specific_nms':
                    selected, selected_scores = model_nms_utils.class_specific_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        box_labels=final_dict['pred_labels'], nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.NMS_CONFIG.get('SCORE_THRESH', None)
                    )
                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms':
                    raise NotImplementedError

                final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                final_dict['pred_scores'] = selected_scores
                final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def crop_gt_features(self, x, gt_boxes, gt_ids, aug_matrix, cfg):
        bev_boxes = []
        bev_box_ids = []
        batch_ids = []
        flip_flags = []
        for i, (boxes_per_batch, ids_per_batch, matrix) in enumerate(zip(gt_boxes, gt_ids, aug_matrix)):
            boxes_per_batch = boxes_per_batch[ids_per_batch != '']
            bev_boxes.append(boxes_per_batch[:, [0, 1, 3, 4, 6]])
            bev_box_ids.append(ids_per_batch[ids_per_batch != ''])
            batch_ids.append(boxes_per_batch.new_full((boxes_per_batch.shape[0], ), i))
            flip_flag = (matrix[0, 0] * matrix[1, 1]) < 0
            flip_flags.append(boxes_per_batch.new_full(
                (boxes_per_batch.shape[0], ), flip_flag, dtype=torch.bool))
        bev_boxes = torch.cat(bev_boxes, dim=0)
        bev_box_ids = np.concatenate(bev_box_ids, axis=0)
        batch_ids = torch.cat(batch_ids, dim=0)
        flip_flags = torch.cat(flip_flags, dim=0)

        bev_boxes[:, 0] = (bev_boxes[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        bev_boxes[:, 1] = (bev_boxes[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        bev_boxes[:, 2] = bev_boxes[:, 2] / self.voxel_size[0]
        bev_boxes[:, 3] = bev_boxes[:, 3] / self.voxel_size[1]
        bev_rois = torch.cat([batch_ids[..., None], bev_boxes], dim=1)
        gt_features = self.roi_align(x, bev_rois)

        # norm gt features to avoid shrink feature scale
        if cfg.get('NORM_FEATURE', False):
            N, C, H, W = gt_features.shape
            gt_features = gt_features.permute(1, 0, 2, 3).reshape(C, -1)
            mean = gt_features.mean(dim=-1, keepdim=True)
            std = gt_features.std(dim=-1, keepdim=True)
            gt_features = (gt_features - mean) / (std + 1e-6)
            gt_features = gt_features.reshape(C, N, H, W).permute(1, 0, 2, 3)

        # flip the roi features when the point cloud has been flipped in dataloader.
        if cfg.get('FLIP_WITH_AUG', False):
            gt_features_flip = torch.flip(gt_features, (2, ))
            gt_features = torch.where(flip_flags[:, None, None, None], gt_features_flip, gt_features)

        feat_size = x.shape[1] * self.roi_align.output_size ** 2
        gt_features = gt_features.reshape(bev_rois.shape[0], feat_size)
        return gt_features, bev_box_ids, batch_ids

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None),
                gt_ids=data_dict.get('gt_ids', None),
            )
            target_dict['lidar_aug_matrix'] = data_dict.get('lidar_aug_matrix', None)
            self.forward_ret_dict['target_dicts'] = target_dict

            if self.model_cfg.get('GT_MATCHING_CFG', None) is not None and \
                self.model_cfg.GT_MATCHING_CFG.get('ENABLE', False):
                gt_features, bev_box_ids, batch_ids = self.crop_gt_features(
                    x, data_dict['gt_boxes'], data_dict['gt_ids'],
                    data_dict['lidar_aug_matrix'], self.model_cfg.GT_MATCHING_CFG)
                self.forward_ret_dict['gt_matching_dict'] = dict(
                    gt_features=gt_features, gt_ids=bev_box_ids, batch_ids=batch_ids)

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict
