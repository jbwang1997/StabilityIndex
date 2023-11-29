import torch
import copy
import pickle
import argparse
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou_bev, boxes_aligned_iou3d_gpu, boxes_iou3d_gpu
from torch_scatter import scatter_max
from tqdm import tqdm
from tabulate import tabulate
import datetime

MAX_IOU_BATCH = 100000


def get_localization_variations(cur_biases, pre_biases, gts):
    cur_biases = torch.from_numpy(cur_biases).float().cuda() 
    pre_biases = torch.from_numpy(pre_biases).float().cuda()
    trans_gts1 = torch.from_numpy(gts).float().cuda()
    trans_gts1[:, :3] = trans_gts1[:, :3] + cur_biases[:, :3]
    trans_gts2 = torch.from_numpy(gts).float().cuda()
    trans_gts2[:, :3] = trans_gts2[:, :3] + pre_biases[:, :3]

    num_boxes = gts.shape[0]
    ious = []
    for start in range(0, num_boxes, MAX_IOU_BATCH):
        end = min(start + MAX_IOU_BATCH, num_boxes)
        ious.append(boxes_aligned_iou3d_gpu(trans_gts1[start:end], trans_gts2[start:end]))
    
    ious = torch.cat(ious)
    ious[ious > 1] = 1
    assert ious.shape[0] == num_boxes
    return ious.cpu().numpy()[:, 0]


def get_extent_variations(cur_biases, pre_biases, gts):
    cur_biases = torch.from_numpy(cur_biases).float().cuda() 
    pre_biases = torch.from_numpy(pre_biases).float().cuda()
    trans_gts1 = torch.from_numpy(gts).float().cuda()
    trans_gts1[:, 3:6] = trans_gts1[:, 3:6] * cur_biases[:, 3:6]
    trans_gts2 = torch.from_numpy(gts).float().cuda()
    trans_gts2[:, 3:6] = trans_gts2[:, 3:6] * pre_biases[:, 3:6]

    num_boxes = gts.shape[0]
    ious = []
    for start in range(0, num_boxes, MAX_IOU_BATCH):
        end = min(start + MAX_IOU_BATCH, num_boxes)
        ious.append(boxes_aligned_iou3d_gpu(trans_gts1[start:end], trans_gts2[start:end]))
    
    ious = torch.cat(ious)
    ious[ious > 1] = 1
    assert ious.shape[0] == num_boxes
    return ious.cpu().numpy()[:, 0]


def get_heading_variations(cur_biases, pre_biases, gts):
    cur_biases = torch.from_numpy(cur_biases).float().cuda() 
    pre_biases = torch.from_numpy(pre_biases).float().cuda()
    trans_gts1 = torch.from_numpy(gts).float().cuda()
    trans_gts1[:, 6] = trans_gts1[:, 6] + cur_biases[:, 6]
    trans_gts2 = torch.from_numpy(gts).float().cuda()
    trans_gts2[:, 6] = trans_gts2[:, 6] + pre_biases[:, 6]

    num_boxes = gts.shape[0]
    ious = []
    for start in range(0, num_boxes, MAX_IOU_BATCH):
        end = min(start + MAX_IOU_BATCH, num_boxes)
        ious.append(boxes_aligned_iou3d_gpu(trans_gts1[start:end], trans_gts2[start:end]))
    
    ious = torch.cat(ious)
    ious[ious > 1] = 1

    # force ious = 0 when angle variaton is larger than np.pi / 4
    angle_vars = torch.abs(cur_biases[:, 6] - pre_biases[:, 6]) % (2 * np.pi)
    angle_vars = torch.minimum(angle_vars, 2*np.pi - angle_vars)
    ious[angle_vars > np.pi/4] = 0

    assert ious.shape[0] == num_boxes
    return ious.cpu().numpy()[:, 0]


def align_bias_into_horizon(det_boxes, gt_boxes):
    det_boxes = torch.from_numpy(det_boxes).float().cuda()
    gt_boxes = torch.from_numpy(gt_boxes).float().cuda()
    Cos, Sin = torch.cos(gt_boxes[:, 6]), torch.sin(gt_boxes[:, 6])
    One, Zero = torch.ones_like(Cos), torch.zeros_like(Cos)
    rot_matrix = torch.stack([Cos, Sin, Zero, 
                              -Sin, Cos, Zero, 
                              Zero, Zero, One], dim=-1).view(-1, 3, 3)
    loc_bias = det_boxes[:, :3] - gt_boxes[:, :3]
    loc_bias = torch.bmm(rot_matrix, loc_bias[:, :, None]).squeeze(-1)
    extent_bias = det_boxes[:, 3:6] / gt_boxes[:, 3:6]
    heading_bias = det_boxes[:, [6]] - gt_boxes[:, [6]]
    return torch.cat([loc_bias, extent_bias, heading_bias], dim=1).cpu().numpy()


def align_det_and_gt_by_hungarian(det_boxes, det_scores, det_names, gt_boxes, gt_names, class_names):
    aligned_boxes, aligned_scores = np.zeros(gt_boxes.shape), np.zeros(gt_boxes.shape[0])
    for i, class_name in enumerate(class_names):
        cls_det_boxes = det_boxes[det_names == class_name] if det_names.shape[0] > 0 else det_boxes
        cls_det_scores = det_scores[det_names == class_name] if det_names.shape[0] > 0 else det_scores
        cls_gt_boxes = gt_boxes[gt_names == class_name] if gt_names.shape[0] > 0 else gt_boxes
        num_det = cls_det_boxes.shape[0]
        num_gt = cls_gt_boxes.shape[0]

        cls_det_boxes = np.concatenate([cls_det_boxes, cls_gt_boxes], axis=0)
        cls_det_scores = np.concatenate([cls_det_scores, np.zeros(cls_gt_boxes.shape[0])], axis=0)

        if cls_gt_boxes.shape[0] == 0:
            continue

        cuda_det_boxes = torch.from_numpy(cls_det_boxes).cuda().float()
        cuda_gt_boxes = torch.from_numpy(cls_gt_boxes).cuda().float()
        ious = boxes_iou3d_gpu(cuda_det_boxes, cuda_gt_boxes).clip(0, 1)
        ious[num_det:, :] = torch.eye(num_gt, dtype=ious.dtype, device=ious.device) * 0.1

        # cost = -(ious.cpu().numpy() * cls_det_scores[:, None])
        cost = -ious.cpu().numpy()
        cost = cost.T
        row_ind, col_ind = linear_sum_assignment(cost)
        aligned_boxes[gt_names == class_name] = cls_det_boxes[col_ind]
        aligned_scores[gt_names == class_name] = cls_det_scores[col_ind]
    
    return aligned_boxes, aligned_scores


def generate_gt_box_mask(info, class_names):
    box_mask = np.array([n in class_names for n in info['name']], dtype=np.bool_)
    zero_difficulty_mask = info['difficulty'] == 0
    info['difficulty'][(info['num_points_in_gt'] > 5) & zero_difficulty_mask] = 1
    info['difficulty'][(info['num_points_in_gt'] <= 5) & zero_difficulty_mask] = 2
    nonzero_mask = info['num_points_in_gt'] > 0
    box_mask = box_mask & nonzero_mask
    return box_mask


def eval_waymo_stable_index(cur_det_annos, pre_det_annos, cur_gt_annos, pre_gt_annos, class_names):
    assert len(cur_det_annos) == len(pre_det_annos) == len(cur_gt_annos) == len(pre_gt_annos), \
        'The number of gt and pred should be aligned.'

    cur_det_annos = copy.deepcopy(cur_det_annos)
    pre_det_annos = copy.deepcopy(pre_det_annos)
    cur_gt_annos = copy.deepcopy(cur_gt_annos)
    pre_gt_annos = copy.deepcopy(pre_gt_annos)

    paired_infos = defaultdict(list)
    for idx in tqdm(range(len(cur_gt_annos))):
        cur_gt_anno, pre_gt_anno = cur_gt_annos[idx], pre_gt_annos[idx]
        cur_det_anno, pre_det_anno = cur_det_annos[idx], pre_det_annos[idx]

        # prepare gt boxes3d
        cur_box_mask = generate_gt_box_mask(cur_gt_anno, class_names)
        pre_box_mask = generate_gt_box_mask(pre_gt_anno, class_names)

        cur_gt_obj_ids = cur_gt_anno['obj_ids'][cur_box_mask]
        pre_gt_obj_ids = pre_gt_anno['obj_ids'][pre_box_mask]
        if cur_gt_obj_ids.shape[0] == 0 or pre_gt_obj_ids.shape[0] == 0:
            continue
        cur_align_idx, pre_align_idx = np.nonzero(
            cur_gt_obj_ids[:, None] == pre_gt_obj_ids[None, :])

        # sample gt boxes3d
        cur_gt_boxes3d = cur_gt_anno['gt_boxes_lidar'][cur_box_mask][cur_align_idx]
        pre_gt_boxes3d = pre_gt_anno['gt_boxes_lidar'][pre_box_mask][pre_align_idx]
        assert cur_gt_boxes3d.shape[0] == pre_gt_boxes3d.shape[0]
        if cur_gt_boxes3d.shape[0] == 0:
            continue

        if cur_gt_boxes3d.shape[-1] == 9:
            cur_gt_boxes3d = cur_gt_boxes3d[:, :7]
        if pre_gt_boxes3d.shape[-1] == 9:
            pre_gt_boxes3d = pre_gt_boxes3d[:, :7]
        # sample name
        gt_names = cur_gt_anno['name'][cur_box_mask][cur_align_idx]
        # sample difficulty
        cur_difficulty = cur_gt_anno['difficulty'][cur_box_mask][cur_align_idx]
        pre_difficulty = pre_gt_anno['difficulty'][pre_box_mask][pre_align_idx]
        difficulty = np.maximum(cur_difficulty, pre_difficulty)
        # sample num_points_in_gt
        cur_num_points = cur_gt_anno['num_points_in_gt'][cur_box_mask][cur_align_idx]
        pre_num_points = pre_gt_anno['num_points_in_gt'][pre_box_mask][pre_align_idx]
        num_points_in_gt = ((cur_num_points + pre_num_points) / 2).astype(np.int64)

        # prepare detected boxes3d
        pre_det_names = pre_det_anno['name']
        pre_det_scores = pre_det_anno['score']
        cur_det_boxes3d = cur_det_anno['boxes_lidar']
        if cur_det_boxes3d.shape[-1] == 9:
            cur_det_boxes3d = cur_det_boxes3d[:, :7]
        cur_det_names = cur_det_anno['name']
        cur_det_scores = cur_det_anno['score']
        pre_det_boxes3d = pre_det_anno['boxes_lidar']
        if pre_det_boxes3d.shape[-1] == 9:
            pre_det_boxes3d = pre_det_boxes3d[:, :7]

        # align prediction and gt by iou
        cur_det_boxes3d, cur_det_scores = align_det_and_gt_by_hungarian(
            cur_det_boxes3d, cur_det_scores, cur_det_names, cur_gt_boxes3d, gt_names, class_names)
        pre_det_boxes3d, pre_det_scores = align_det_and_gt_by_hungarian(
            pre_det_boxes3d, pre_det_scores, pre_det_names, pre_gt_boxes3d, gt_names, class_names)

        paired_infos['cur_gt_boxes3d'].append(cur_gt_boxes3d)
        paired_infos['pre_gt_boxes3d'].append(pre_gt_boxes3d)
        paired_infos['difficult'].append(difficulty)
        paired_infos['num_points_in_gt'].append(num_points_in_gt)
        paired_infos['cur_det_boxes3d'].append(cur_det_boxes3d)
        paired_infos['cur_det_scores'].append(cur_det_scores)
        paired_infos['pre_det_boxes3d'].append(pre_det_boxes3d)
        paired_infos['pre_det_scores'].append(pre_det_scores)
        paired_infos['gt_names'].append(gt_names)

    paired_infos = dict(paired_infos)
    for key, value in paired_infos.items():
        paired_infos[key] = np.concatenate(value)

    # fliter out not detected objects
    not_det_mask = (paired_infos['pre_det_scores'] == 0) & (paired_infos['cur_det_scores'] == 0)
    for key, value in paired_infos.items():
        paired_infos[key] = value[~not_det_mask]
    
    cur_gts, pre_gts = paired_infos['cur_gt_boxes3d'], paired_infos['pre_gt_boxes3d']
    cur_dets, pre_dets = paired_infos['cur_det_boxes3d'], paired_infos['pre_det_boxes3d']
    nboxes = cur_gts.shape[0]
    
    # norm ground truth
    norm_gts = np.zeros((nboxes, 7))
    norm_gts[:, 3:6] = np.sqrt(cur_gts[:, 3:6] * pre_gts[:, 3:6])

    # alignment bias into normalized car coordinates
    cur_det_biases = align_bias_into_horizon(cur_dets, cur_gts)
    pre_det_biases = align_bias_into_horizon(pre_dets, pre_gts)
    
    # calculate stable index
    metrics = dict()
    confidence_vars = paired_infos['cur_det_scores'] - paired_infos['pre_det_scores']
    confidence_vars = confidence_vars / (np.quantile(paired_infos['cur_det_scores'], 0.99) - \
        np.quantile(paired_infos['cur_det_scores'], 0.01))
    localization_vars = get_localization_variations(cur_det_biases, pre_det_biases, norm_gts)
    extent_vars = get_extent_variations(cur_det_biases, pre_det_biases, norm_gts)
    heading_vars = get_heading_variations(cur_det_biases, pre_det_biases, norm_gts)

    stable_index = (1 - np.abs(confidence_vars)) * (localization_vars + extent_vars + heading_vars) / 3

    paired_infos['confidence_vars'] = confidence_vars
    paired_infos['localization_vars'] = localization_vars
    paired_infos['extent_vars'] = extent_vars
    paired_infos['heading_vars'] = heading_vars
    paired_infos['stable_index'] = stable_index

    def get_values_by_mask(*args, mask=None):
        assert mask is not None
        return [arg[mask] for arg in args]

    distances = np.linalg.norm(paired_infos['cur_det_boxes3d'][:, :3], axis=1)
    for class_name in class_names:
        class_mask = paired_infos['gt_names'] == class_name
        _confidence, _localization, _extent, _heading, _stable_index = get_values_by_mask(
            confidence_vars, localization_vars, extent_vars, heading_vars, stable_index, mask=class_mask)
        metrics['CONFIDENCE_VARIATION_%s' % class_name] = (1 - np.abs(_confidence)).mean()
        metrics['LOCALIZATION_VARIATION_%s' % class_name] = _localization.mean()
        metrics['EXTENT_VARIATION_%s' % class_name] = _extent.mean()
        metrics['HEADING_VARIATION_%s' % class_name] = _heading.mean()
        metrics['STABLE_INDEX_%s' % class_name] = _stable_index.mean()
        
        for idx, distance in enumerate([[0, 30], [30, 50], [50, np.float('inf')]]):
            MIN, MAX = distance
            distance_mask = (distances > MIN) & (distances <= MAX)
            cur_mask = class_mask & distance_mask
            _confidence, _localization, _extent, _heading, _stable_index = get_values_by_mask(
                confidence_vars, localization_vars, extent_vars, heading_vars, stable_index, mask=cur_mask)
            metrics['CONFIDENCE_VARIATION_%s_%s_to_%s' % (class_name, MIN, MAX)] = (1 - np.abs(_confidence)).mean()
            metrics['LOCALIZATION_VARIATION_%s_%s_to_%s' % (class_name, MIN, MAX)] = _localization.mean()
            metrics['EXTENT_VARIATION_%s_%s_to_%s' % (class_name, MIN, MAX)] = _extent.mean()
            metrics['HEADING_VARIATION_%s_%s_to_%s' % (class_name, MIN, MAX)] = _heading.mean()
            metrics['STABLE_INDEX_%s_%s_to_%s' % (class_name, MIN, MAX)] = _stable_index.mean()
    return metrics


def print_stable_index_results(metrics, class_names):
    metrics_str = ''
    metrics_str += '\n----------------------------------\n'
    metrics_str += f'Stable Index (Overall)'
    metrics_str += '\n----------------------------------\n'

    metrics_data_print = []
    for class_name in class_names:
        metrics_data_print.append([class_name, 
                                   f"{metrics['STABLE_INDEX_%s' % class_name]:.4f}", 
                                   f"{metrics['CONFIDENCE_VARIATION_%s' % class_name]:.4f}",
                                   f"{metrics['LOCALIZATION_VARIATION_%s' % class_name]:.4f}",
                                   f"{metrics['EXTENT_VARIATION_%s' % class_name]:.4f}",
                                   f"{metrics['HEADING_VARIATION_%s' % class_name]:.4f}"])
    metrics_str += tabulate(metrics_data_print, headers=['class', 'SI', 'confidence', 
                                                        'localization', 'extent',
                                                        'heading'], tablefmt='orgtbl')

    # print distance breakdown    
    metrics_data_print = []                                                    
    metrics_str += '\n\n----------------------------------\n'
    metrics_str += f'Stable Index (BreakDown By Distance)'
    metrics_str += '\n----------------------------------\n'

    for class_name in class_names:
        for idx, distance in enumerate([[0, 30], [30, 50], [50, np.float('inf')]]):
            MIN, MAX = distance
            metrics_data_print.append([class_name if idx == 0 else '', 
                            ['<30', '30-50', '>50'][idx],
                            f"{metrics['STABLE_INDEX_%s_%s_to_%s' % (class_name, MIN, MAX)]:.4f}",
                            f"{metrics['CONFIDENCE_VARIATION_%s_%d_to_%s' % (class_name, MIN, MAX)]:.4f}",
                            f"{metrics['LOCALIZATION_VARIATION_%s_%d_to_%s' % (class_name, MIN, MAX)]:.4f}",
                            f"{metrics['EXTENT_VARIATION_%s_%s_to_%s' % (class_name, MIN, MAX)]:.4f}",
                            f"{metrics['HEADING_VARIATION_%s_%s_to_%s' % (class_name, MIN, MAX)]:.4f}"])
        metrics_data_print += '\n'
    metrics_str += tabulate(metrics_data_print, headers=['class', 'dist.', 'SI', 'confidence', 'localization', 
                                                         'extent', 'heading'], tablefmt='orgtbl')                            


    return metrics_str


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--gt_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Vehicle', 'Pedestrian', 'Cyclist'], help='')
    parser.add_argument('--sampled_interval', type=int, default=5, help='sampled interval for GT sequences')
    args = parser.parse_args()

    pred_infos = pickle.load(open(args.pred_infos, 'rb'))
    gt_infos = pickle.load(open(args.gt_infos, 'rb'))

    frame_id_mapper = {info['frame_id']: i for i, info in enumerate(gt_infos)}
    cur_det_annos, pre_det_annos = [], []
    cur_gt_annos, pre_gt_annos = [], []
    for i, info in enumerate(gt_infos):
        sequence_name = info['point_cloud']['lidar_sequence']
        sample_idx = info['point_cloud']['sample_idx']
        # filter out frame withous previous information
        pre_sample_idx = sample_idx - 5
        pre_frame_idx = sequence_name + '_%03d' % (pre_sample_idx, )
        if pre_frame_idx not in frame_id_mapper:
            continue

        cur_det_annos.append(pred_infos[i])
        cur_gt_annos.append(gt_infos[i]['annos'])
        pre_det_annos.append(pred_infos[frame_id_mapper[pre_frame_idx]])
        pre_gt_annos.append(gt_infos[frame_id_mapper[pre_frame_idx]]['annos'])

    stable_index = eval_waymo_stable_index(cur_det_annos, pre_det_annos, cur_gt_annos, pre_gt_annos, args.class_names)
    stable_index_str = print_stable_index_results(stable_index, args.class_names)
    print(stable_index_str)

    # save to file
    log_file = '/'.join(args.pred_infos.split('/')[:-1]) + f'/eval_si_%s.log' %datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    print('Save log file to %s' % log_file)
    with open(log_file, 'w') as f:
        f.write(stable_index_str)

if __name__ == '__main__':
    main()
