import torch
import copy
import pickle
import argparse
import numpy as np
from collections import defaultdict
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou_bev, boxes_aligned_iou3d_gpu, boxes_iou3d_gpu
from torch_scatter import scatter_max
from tqdm import tqdm
from tabulate import tabulate
import datetime

MAX_IOU_BATCH = 100000


def get_translation_variation(det_boxes1, det_boxes2, gt_boxes1, gt_boxes2):
    det_boxes1 = torch.from_numpy(det_boxes1).float().cuda()
    det_boxes2 = torch.from_numpy(det_boxes2).float().cuda()

    translations = gt_boxes2[:, :3] - gt_boxes1[:, :3]
    translations = torch.from_numpy(translations).float().cuda()
    det_boxes1_ = det_boxes1.clone()
    det_boxes1_[:, :3] = det_boxes1_[:, :3] + translations
    det_boxes1_[:, 3:] = det_boxes2[:, 3:]
    det_boxes2_ = det_boxes2.clone()
    det_boxes2_[:, :3] = det_boxes2_[:, :3] - translations
    det_boxes2_[:, 3:] = det_boxes1[:, 3:]

    num_boxes = det_boxes1.shape[0]
    ious = []
    for start_idx in range(0, num_boxes, MAX_IOU_BATCH):
        end_idx = min(start_idx + MAX_IOU_BATCH, num_boxes)
        ious1 = boxes_aligned_iou3d_gpu(
            det_boxes1_[start_idx:end_idx], det_boxes2[start_idx:end_idx])
        ious1[ious1 > 1] = 1
        ious2 = boxes_aligned_iou3d_gpu(
            det_boxes1[start_idx:end_idx], det_boxes2_[start_idx:end_idx])
        ious2[ious2 > 1] = 1
        ious.append((ious1 + ious2) / 2)
    
    ious = torch.cat(ious)
    assert ious.shape[0] == num_boxes
    return ious.cpu().numpy()[:, 0]


def get_size_variation(det_boxes1, det_boxes2, gt_boxes1, gt_boxes2):
    det_boxes1 = torch.from_numpy(det_boxes1).float().cuda()
    det_boxes2 = torch.from_numpy(det_boxes2).float().cuda()

    scales = gt_boxes2[:, 3:6] / gt_boxes1[:, 3:6]
    scales = torch.from_numpy(scales).float().cuda()
    det_boxes1_ = det_boxes1.clone()
    det_boxes1_[:, 3:6] = det_boxes1_[:, 3:6] * scales
    det_boxes1_[:, [0, 1, 2, 6]] = det_boxes2[:, [0, 1, 2, 6]] 
    det_boxes2_ = det_boxes2.clone()
    det_boxes2_[:, 3:6] = det_boxes2_[:, 3:6] / scales
    det_boxes2_[:, [0, 1, 2, 6]] = det_boxes1[:, [0, 1, 2, 6]] 

    num_boxes = det_boxes1.shape[0]
    ious = []
    for start_idx in range(0, num_boxes, MAX_IOU_BATCH):
        end_idx = min(start_idx + MAX_IOU_BATCH, num_boxes)
        ious1 = boxes_aligned_iou3d_gpu(
            det_boxes1_[start_idx:end_idx], det_boxes2[start_idx:end_idx])
        ious1[ious1 > 1] = 1
        ious2 = boxes_aligned_iou3d_gpu(
            det_boxes1[start_idx:end_idx], det_boxes2_[start_idx:end_idx])
        ious2[ious2 > 1] = 1
        ious.append((ious1 + ious2) / 2)
    
    ious = torch.cat(ious)
    assert ious.shape[0] == num_boxes
    return ious.cpu().numpy()[:, 0]


def get_heading_variation(det_boxes1, det_boxes2, gt_boxes1, gt_boxes2):
    det_boxes1 = torch.from_numpy(det_boxes1).float().cuda()
    mask = (det_boxes1[:, 3:5].max(dim=1)[0] / det_boxes1[:, 3:5].min(dim=1)[0]) < 2
    det_boxes1[mask, 3] = torch.where(
        det_boxes1[mask, 3] > det_boxes1[mask, 4],
        det_boxes1[mask, 4] * 2, det_boxes1[mask, 3])
    det_boxes1[mask, 4] = torch.where(
        det_boxes1[mask, 3] > det_boxes1[mask, 4],
        det_boxes1[mask, 4], det_boxes1[mask, 3] * 2)

    det_boxes2 = torch.from_numpy(det_boxes2).float().cuda()
    mask = (det_boxes2[:, 3:5].max(dim=1)[0] / det_boxes2[:, 3:5].min(dim=1)[0]) < 2
    det_boxes2[mask, 3] = torch.where(
        det_boxes2[mask, 3] > det_boxes2[mask, 4],
        det_boxes2[mask, 4] * 2, det_boxes2[mask, 3])
    det_boxes2[mask, 4] = torch.where(
        det_boxes2[mask, 3] > det_boxes2[mask, 4],
        det_boxes2[mask, 4], det_boxes2[mask, 3] * 2)

    rotation = gt_boxes2[:, 6] - gt_boxes1[:, 6]
    rotation = torch.from_numpy(rotation).float().cuda()
    det_boxes1_ = det_boxes2.clone()
    det_boxes1_[:, 6] = det_boxes1[:, 6] + rotation
    det_boxes2_ = det_boxes1.clone()
    det_boxes2_[:, 6] = det_boxes2[:, 6] - rotation

    num_boxes = det_boxes1.shape[0]
    ious = []
    for start_idx in range(0, num_boxes, MAX_IOU_BATCH):
        end_idx = min(start_idx + MAX_IOU_BATCH, num_boxes)
        ious1 = boxes_aligned_iou3d_gpu(
            det_boxes1_[start_idx:end_idx], det_boxes2[start_idx:end_idx])
        ious1[ious1 > 1] = 1
        ious2 = boxes_aligned_iou3d_gpu(
            det_boxes1[start_idx:end_idx], det_boxes2_[start_idx:end_idx])
        ious2[ious2 > 1] = 1
        ious.append((ious1 + ious2) / 2)
    
    ious = torch.cat(ious)
    assert ious.shape[0] == num_boxes
    return ious.cpu().numpy()[:, 0]


def align_det_and_gt_by_max_iou(det_boxes, det_scores, det_names, gt_boxes, gt_names, class_names):
    if det_boxes.shape[0] == 0:
        return copy.deepcopy(gt_boxes), np.zeros(gt_boxes.shape[0])

    det_boxes_ = copy.deepcopy(det_boxes)
    gt_boxes_ = copy.deepcopy(gt_boxes)
    for i, class_name in enumerate(class_names):
        det_mask = det_names == class_name
        det_boxes_[det_mask][:, :2] = det_boxes_[det_mask][:, :2] + 1000 * i
        gt_mask = gt_names == class_name
        gt_boxes_[gt_mask][:, :2] = gt_boxes_[gt_mask][:, :2] + 1000 * i
    
    gt_boxes_ = torch.from_numpy(gt_boxes_).float().cuda()
    det_boxes_ = torch.from_numpy(det_boxes_).float().cuda()
    ious = boxes_iou3d_gpu(gt_boxes_, det_boxes_)
    max_iou, max_idx = torch.max(ious, axis=1)
    max_iou = max_iou.cpu().numpy()
    max_idx = max_idx.cpu().numpy()

    aligned_boxes = det_boxes[max_idx]
    aligned_scores = det_scores[max_idx]
    # if the max_iou < 0.3, we think this gt has no aligned detection boxes.
    if (max_iou < 0.1).any():
        aligned_scores[max_iou < 0.1] = 0
        aligned_boxes[max_iou < 0.1] = gt_boxes[max_iou < 0.1]

    return aligned_boxes, aligned_scores


def align_det_and_gt_by_max_score(det_boxes, det_scores, det_names, gt_boxes, gt_names, class_names):
    if det_boxes.shape[0] == 0:
        return copy.deepcopy(gt_boxes), np.zeros(gt_boxes.shape[0])

    gt_boxes_ = torch.from_numpy(gt_boxes).float().cuda()
    det_boxes_ = torch.from_numpy(det_boxes).float().cuda()
    for i, class_name in enumerate(class_names):
        det_mask = det_names == class_name
        det_boxes_[det_mask][:, :2] = det_boxes_[det_mask][:, :2] + 1000 * i
        gt_mask = gt_names == class_name
        gt_boxes_[gt_mask][:, :2] = gt_boxes_[gt_mask][:, :2] + 1000 * i
    det_scores_ = torch.from_numpy(det_scores).float().cuda()

    ious = boxes_iou3d_gpu(det_boxes_, gt_boxes_)
    ious = ious.clip(0, 1)
    max_iou, max_idx = torch.max(ious, axis=1)

    det_boxes = det_boxes[max_iou.cpu().numpy() > 0.1]
    if det_boxes.shape[0] == 0:
        return copy.deepcopy(gt_boxes), np.zeros(gt_boxes.shape[0])
    det_scores_ = det_scores_[max_iou > 0.1]
    max_idx = max_idx[max_iou > 0.1]
    max_iou = max_iou[max_iou > 0.1]

    num_gt, num_tp = gt_boxes.shape[0], max_iou.shape[0]
    scatter_max_scores = torch.zeros(num_gt, dtype=torch.float32).cuda()
    scatter_max_scores, scatter_max_idx = scatter_max(
        det_scores_, max_idx, out=scatter_max_scores)
    fn_mask = (scatter_max_idx < 0) | (scatter_max_idx > num_tp - 1)
    scatter_max_idx = scatter_max_idx.clip(0, num_tp - 1)

    aligned_idx = scatter_max_idx.cpu().numpy()
    fn_mask = fn_mask.cpu().numpy()
    aligned_boxes = det_boxes[aligned_idx]
    aligned_boxes[fn_mask] = gt_boxes[fn_mask]
    aligned_scores = scatter_max_scores.cpu().numpy()
    return aligned_boxes, aligned_scores


def generate_gt_box_mask(info, class_names):
    box_mask = np.array([n in class_names for n in info['name']], dtype=np.bool_)
    zero_difficulty_mask = info['difficulty'] == 0
    info['difficulty'][(info['num_points_in_gt'] > 5) & zero_difficulty_mask] = 1
    info['difficulty'][(info['num_points_in_gt'] <= 5) & zero_difficulty_mask] = 2
    nonzero_mask = info['num_points_in_gt'] > 0
    box_mask = box_mask & nonzero_mask
    return box_mask


def eval_stable_index(cur_det_annos, pre_det_annos, cur_gt_annos, 
                      pre_gt_annos, class_names, align_func='max_iou'):
    num_frame_pair = len(cur_det_annos)
    assert len(pre_det_annos) == num_frame_pair
    assert len(cur_gt_annos) == num_frame_pair
    assert len(pre_gt_annos) == num_frame_pair
    assert align_func in ['max_iou', 'max_score']
    align_func = align_det_and_gt_by_max_iou if align_func == 'max_iou' \
        else align_det_and_gt_by_max_score

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
        cur_det_boxes3d, cur_det_scores = align_func(cur_det_boxes3d, cur_det_scores, cur_det_names,
                                                     cur_gt_boxes3d, gt_names, class_names)
        pre_det_boxes3d, pre_det_scores = align_func(pre_det_boxes3d, pre_det_scores, pre_det_names,
                                                     pre_gt_boxes3d, gt_names, class_names)

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

    not_det_mask = (paired_infos['pre_det_scores'] == 0) & (paired_infos['cur_det_scores'] == 0)
    for key, value in paired_infos.items():
        paired_infos[key] = value[~not_det_mask]
    
    # calculate stable index
    metrics = dict()
    score_diffs = paired_infos['cur_det_scores'] - paired_infos['pre_det_scores']
    score_diffs = score_diffs / (np.quantile(paired_infos['cur_det_scores'], 0.99) - \
        np.quantile(paired_infos['cur_det_scores'], 0.01))

    trans_diffs = get_translation_variation(paired_infos['pre_det_boxes3d'],
                                            paired_infos['cur_det_boxes3d'],
                                            paired_infos['pre_gt_boxes3d'],
                                            paired_infos['cur_gt_boxes3d'])
    size_diffs = get_size_variation(paired_infos['pre_det_boxes3d'],
                                    paired_infos['cur_det_boxes3d'],
                                    paired_infos['pre_gt_boxes3d'],
                                    paired_infos['cur_gt_boxes3d'])
    heading_diffs = get_heading_variation(paired_infos['pre_det_boxes3d'],
                                          paired_infos['cur_det_boxes3d'],
                                          paired_infos['pre_gt_boxes3d'],
                                          paired_infos['cur_gt_boxes3d'])
    stable_index = (1 - np.abs(score_diffs)) * (trans_diffs + size_diffs + heading_diffs) / 3

    def get_values_by_mask(score_diffs, trans_diffs, size_diffs, heading_diffs, stable_index, mask):
        return score_diffs[mask], trans_diffs[mask], size_diffs[mask], heading_diffs[mask], stable_index[mask]

    metrics['DISTANCE_BREAKDOWN'] = {}
    distances = np.linalg.norm(paired_infos['cur_det_boxes3d'][:, :3], axis=1)
    
    for class_name in class_names:
        class_mask = paired_infos['gt_names'] == class_name
        _score, _trans, _size, _heading, _stable_index = get_values_by_mask(
            score_diffs, trans_diffs, size_diffs, heading_diffs, stable_index, class_mask)
        metrics['SCORE_VARIATION_%s' % class_name] = (1 - np.abs(_score)).mean()
        metrics['TRANSLATION_VARIATION_%s' % class_name] = _trans.mean()
        metrics['SIZE_VARIATION_%s' % class_name] = _size.mean()
        metrics['HEADING_VARIATION_%s' % class_name] = _heading.mean()
        metrics['STABLE_INDEX_%s' % class_name] = _stable_index.mean()
        
        for idx, distance in enumerate([[0, 30], [30, 50], [50, np.float('inf')]]):
            distance_mask = (distances > distance[0]) & (distances <= distance[1])
            cur_mask = class_mask & distance_mask
            _score, _trans, _size, _heading, _stable_index = get_values_by_mask(
                score_diffs, trans_diffs, size_diffs, heading_diffs, stable_index, cur_mask)
            metrics['DISTANCE_BREAKDOWN']['SCORE_VARIATION_%s_%d' % (class_name, idx)] = (1 - np.abs(_score)).mean()
            metrics['DISTANCE_BREAKDOWN']['TRANSLATION_VARIATION_%s_%d' % (class_name, idx)] = _trans.mean()
            metrics['DISTANCE_BREAKDOWN']['SIZE_VARIATION_%s_%d' % (class_name, idx)] = _size.mean()
            metrics['DISTANCE_BREAKDOWN']['HEADING_VARIATION_%s_%d' % (class_name, idx)] = _heading.mean()
            metrics['DISTANCE_BREAKDOWN']['STABLE_INDEX_%s_%d' % (class_name, idx)] = _stable_index.mean()
    return metrics


def print_metrics(metrics, class_names):
    metrics_str = ''
    metrics_str += '\n----------------------------------\n'
    metrics_str += f'Stable Index (Overall)'
    metrics_str += '\n----------------------------------\n'

    metrics_data_print = []
    for class_name in class_names:
        metrics_data_print.append([class_name, metrics['STABLE_INDEX_%s' % class_name], 
                                   f"{metrics['SCORE_VARIATION_%s' % class_name]:.4f}",
                                   f"{metrics['TRANSLATION_VARIATION_%s' % class_name]:.4f}",
                                   f"{metrics['SIZE_VARIATION_%s' % class_name]:.4f}",
                                   f"{metrics['HEADING_VARIATION_%s' % class_name]:.4f}"])
    metrics_str += tabulate(metrics_data_print, headers=['class', 'SI', 'score', 
                                                        'translation', 'size',
                                                        'heading'], tablefmt='orgtbl')

    # print distance breakdown    
    metrics_data_print = []                                                    
    metrics_str += '\n\n----------------------------------\n'
    metrics_str += f'Stable Index (BreakDown By Distance)'
    metrics_str += '\n----------------------------------\n'
    d_metrics = metrics['DISTANCE_BREAKDOWN']

    for class_name in class_names:
        for idx, distance in enumerate([[0, 30], [30, 50], [50, np.float('inf')]]):
            metrics_data_print.append([class_name if idx == 0 else '', 
                            ['<30', '30-50', '>50'][idx],
                            f"{d_metrics['STABLE_INDEX_%s_%d' % (class_name, idx)]:.4f}",
                            f"{d_metrics['SCORE_VARIATION_%s_%d' % (class_name, idx)]:.4f}",
                            f"{d_metrics['TRANSLATION_VARIATION_%s_%d' % (class_name, idx)]:.4f}",
                            f"{d_metrics['SIZE_VARIATION_%s_%d' % (class_name, idx)]:.4f}",
                            f"{d_metrics['HEADING_VARIATION_%s_%d' % (class_name, idx)]:.4f}"])
        metrics_data_print += '\n'
    metrics_str += tabulate(metrics_data_print, headers=['class', 'dist.', 'SI', 'score', 
                                                    'translation', 'size', 'heading'], tablefmt='orgtbl')                            


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

    stable_index = eval_stable_index(cur_det_annos, pre_det_annos, cur_gt_annos, pre_gt_annos, args.class_names)
    stable_index_str = print_metrics(stable_index, args.class_names)
    print(stable_index_str)

    # save to file
    log_file = '/'.join(args.pred_infos.split('/')[:-1]) + f'/eval_si_%s.log' %datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    print('Save log file to %s' % log_file)
    with open(log_file, 'w') as f:
        f.write(stable_index_str)

if __name__ == '__main__':
    main()
