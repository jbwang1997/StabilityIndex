import torch
import copy
import pickle
import argparse
import numpy as np
from collections import defaultdict
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou_bev, boxes_aligned_iou3d_gpu

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


def max_iou_assign_det_and_gt(det_boxes, det_scores, det_names, gt_boxes, gt_names, class_names):
    det_boxes_ = copy.deepcopy(det_boxes)
    gt_boxes_ = copy.deepcopy(gt_boxes)
    for i, class_name in enumerate(class_names):
        det_mask = det_names == class_name
        det_boxes_[det_mask][:, :2] = det_boxes_[det_mask][:, :2] + 1000 * i
        gt_mask = gt_names == class_name
        gt_boxes_[gt_mask][:, :2] = gt_boxes_[gt_mask][:, :2] + 1000 * i
    
    gt_boxes_ = torch.from_numpy(gt_boxes_).float().cuda()
    det_boxes_ = torch.from_numpy(det_boxes_).float().cuda()
    ious = boxes_iou_bev(gt_boxes_, det_boxes_)
    max_iou, max_idx = torch.max(ious, axis=1)
    max_iou = max_iou.cpu().numpy()
    max_idx = max_idx.cpu().numpy()

    aligned_boxes = det_boxes[max_idx]
    aligned_scores = det_scores[max_idx]
    # if the max_iou < 0.3, we think this gt has no aligned detection boxes.
    if (max_iou < 0.3).any():
        aligned_scores[max_iou < 0.3] = 0
        aligned_boxes[max_iou < 0.3] = gt_boxes[max_iou < 0.3]

    return aligned_boxes, aligned_scores


def generate_gt_box_mask(info, class_names):
    box_mask = np.array([n in class_names for n in info['name']], dtype=np.bool_)
    zero_difficulty_mask = info['difficulty'] == 0
    info['difficulty'][(info['num_points_in_gt'] > 5) & zero_difficulty_mask] = 1
    info['difficulty'][(info['num_points_in_gt'] <= 5) & zero_difficulty_mask] = 2
    nonzero_mask = info['num_points_in_gt'] > 0
    box_mask = box_mask & nonzero_mask
    return box_mask


def eval_stable_index(cur_det_annos, pre_det_annos, cur_gt_annos, pre_gt_annos, class_names):
    num_frame_pair = len(cur_det_annos)
    assert len(pre_det_annos) == num_frame_pair
    assert len(cur_gt_annos) == num_frame_pair
    assert len(pre_gt_annos) == num_frame_pair

    cur_det_annos = copy.deepcopy(cur_det_annos)
    pre_det_annos = copy.deepcopy(pre_det_annos)
    cur_gt_annos = copy.deepcopy(cur_gt_annos)
    pre_gt_annos = copy.deepcopy(pre_gt_annos)

    paired_infos = defaultdict(list)
    for cur_det_anno, pre_det_anno, cur_gt_anno, pre_gt_anno in zip(
        cur_det_annos, pre_det_annos, cur_gt_annos, pre_gt_annos):
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
        num_points_in_gt = ((cur_num_points + pre_num_points) / 2).astype(np.int)

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
        if cur_det_boxes3d.shape[0] == 0 or pre_det_boxes3d.shape[0] == 0:
            continue

        # align prediction and gt by iou
        cur_det_boxes3d, cur_det_scores = \
            max_iou_assign_det_and_gt(cur_det_boxes3d, cur_det_scores, cur_det_names,
                                      cur_gt_boxes3d, gt_names, class_names)
        pre_det_boxes3d, pre_det_scores = \
            max_iou_assign_det_and_gt(pre_det_boxes3d, pre_det_scores, pre_det_names,
                                      pre_gt_boxes3d, gt_names, class_names)

        paired_infos['cur_gt_boxes3d'].append(cur_gt_boxes3d)
        paired_infos['pre_gt_boxes3d'].append(pre_gt_boxes3d)
        paired_infos['difficult'].append(difficulty)
        paired_infos['num_points_in_gt'].append(num_points_in_gt)
        paired_infos['cur_det_boxes3d'].append(cur_det_boxes3d)
        paired_infos['cur_det_scores'].append(cur_det_scores)
        paired_infos['pre_det_boxes3d'].append(pre_det_boxes3d)
        paired_infos['pre_det_scores'].append(pre_det_scores)
    
    paired_infos = dict(paired_infos)
    for key, value in paired_infos.items():
        paired_infos[key] = np.concatenate(value)
        
    # calculate stable index
    metrics = dict()
    score_diffs = np.abs(paired_infos['cur_det_scores'] - paired_infos['pre_det_scores'])
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
    stable_index = (1 - score_diffs) * (trans_diffs + size_diffs + heading_diffs) / 3
    metrics['SCORE_VARIATION'] = (1 - score_diffs).mean()
    metrics['TRANSLATION_VARIATION'] = trans_diffs.mean()
    metrics['SIZE_VARIATION'] = size_diffs.mean()
    metrics['HEADING_VARIATION'] = heading_diffs.mean()
    metrics['STABLE_INDEX'] = stable_index.mean()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--gt_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Vehicle', 'Pedestrian', 'Cyclist'], help='')
    parser.add_argument('--sampled_interval', type=int, default=5, help='sampled interval for GT sequences')
    args = parser.parse_args()

    pred_infos = pickle.load(open(args.pred_infos, 'rb'))
    gt_infos = pickle.load(open(args.gt_infos, 'rb'))

    # print('Start to evaluate the waymo format results...')
    # eval = OpenPCDetWaymoDetectionMetricsEstimator()

    # gt_infos_dst = []
    # for idx in range(0, len(gt_infos), args.sampled_interval):
    #     cur_info = gt_infos[idx]['annos']
    #     cur_info['frame_id'] = gt_infos[idx]['frame_id']
    #     gt_infos_dst.append(cur_info)

    # waymo_AP = eval.waymo_evaluation(
    #     pred_infos, gt_infos_dst, class_name=args.class_names, distance_thresh=1000, fake_gt_infos=False
    # )

    # print(waymo_AP)


if __name__ == '__main__':
    main()
