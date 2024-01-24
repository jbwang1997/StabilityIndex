import torch
import copy
import pickle
import argparse
import numpy as np

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_aligned_iou3d_gpu, boxes_iou3d_gpu

from collections import defaultdict
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from tabulate import tabulate
import datetime

MAX_IOU_BATCH = 100000


map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}


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


def eval_nuscenes_stability_index(cur_det_annos, pre_det_annos, cur_gt_annos, pre_gt_annos, class_names=None):
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

        cur_gt_obj_ids = cur_gt_anno['instance_tokens']
        pre_gt_obj_ids = pre_gt_anno['instance_tokens']
        if cur_gt_obj_ids.shape[0] == 0 or pre_gt_obj_ids.shape[0] == 0:
            continue
        cur_align_idx, pre_align_idx = np.nonzero(
            cur_gt_obj_ids[:, None] == pre_gt_obj_ids[None, :])

        # sample gt boxes3d
        cur_gt_boxes3d = cur_gt_anno['boxes'][cur_align_idx]
        pre_gt_boxes3d = pre_gt_anno['boxes'][pre_align_idx]
        assert cur_gt_boxes3d.shape[0] == pre_gt_boxes3d.shape[0]
        if cur_gt_boxes3d.shape[0] == 0:
            continue

        if cur_gt_boxes3d.shape[-1] == 9:
            cur_gt_boxes3d = cur_gt_boxes3d[:, :7]
        if pre_gt_boxes3d.shape[-1] == 9:
            pre_gt_boxes3d = pre_gt_boxes3d[:, :7]
        # sample name
        gt_names = cur_gt_anno['gt_names'][cur_align_idx]
        # sample num_points_in_gt
        cur_num_points = cur_gt_anno['num_lidar_pts'][cur_align_idx]
        pre_num_points = pre_gt_anno['num_lidar_pts'][pre_align_idx]
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
    
    # calculate stability index
    metrics = dict()
    confidence_vars = paired_infos['cur_det_scores'] - paired_infos['pre_det_scores']
    confidence_vars = confidence_vars / (np.quantile(paired_infos['cur_det_scores'], 0.99) - \
        np.quantile(paired_infos['cur_det_scores'], 0.01))
    localization_vars = get_localization_variations(cur_det_biases, pre_det_biases, norm_gts)
    extent_vars = get_extent_variations(cur_det_biases, pre_det_biases, norm_gts)
    heading_vars = get_heading_variations(cur_det_biases, pre_det_biases, norm_gts)

    stability_index = (1 - np.abs(confidence_vars)) * (localization_vars + extent_vars + heading_vars) / 3

    paired_infos['confidence_vars'] = confidence_vars
    paired_infos['localization_vars'] = localization_vars
    paired_infos['extent_vars'] = extent_vars
    paired_infos['heading_vars'] = heading_vars
    paired_infos['stability_index'] = stability_index

    def get_values_by_mask(*args, mask=None):
        assert mask is not None
        return [arg[mask] for arg in args]

    distances = np.linalg.norm(paired_infos['cur_det_boxes3d'][:, :3], axis=1)
    for class_name in class_names:
        class_mask = paired_infos['gt_names'] == class_name
        _confidence, _localization, _extent, _heading, _stability_index = get_values_by_mask(
            confidence_vars, localization_vars, extent_vars, heading_vars, stability_index, mask=class_mask)
        metrics['CONFIDENCE_VARIATION_%s' % class_name] = (1 - np.abs(_confidence)).mean()
        metrics['LOCALIZATION_VARIATION_%s' % class_name] = _localization.mean()
        metrics['EXTENT_VARIATION_%s' % class_name] = _extent.mean()
        metrics['HEADING_VARIATION_%s' % class_name] = _heading.mean()
        metrics['STABILITY_INDEX_%s' % class_name] = _stability_index.mean()
        
    # Calculate mean stability index
    for si_type in ['CONFIDENCE_VARIATION', 'LOCALIZATION_VARIATION', 'EXTENT_VARIATION',
                    'HEADING_VARIATION', 'STABILITY_INDEX']:
        collector = [metrics['%s_%s' % (si_type, class_name)] for class_name in class_names]
        metrics['%s_mean' % si_type] = sum(collector) / len(collector)

    return metrics


def print_stability_index_results(metrics, class_names):
    metrics_str = ''
    metrics_str += '\n----------------------------------\n'
    metrics_str += f'Stability Index (Overall)'
    metrics_str += '\n----------------------------------\n'

    metrics_data_print = []
    for class_name in [*class_names, 'mean']:
        metrics_data_print.append([class_name, 
                                   f"{metrics['STABILITY_INDEX_%s' % class_name]:.4f}", 
                                   f"{metrics['CONFIDENCE_VARIATION_%s' % class_name]:.4f}",
                                   f"{metrics['LOCALIZATION_VARIATION_%s' % class_name]:.4f}",
                                   f"{metrics['EXTENT_VARIATION_%s' % class_name]:.4f}",
                                   f"{metrics['HEADING_VARIATION_%s' % class_name]:.4f}"])
    metrics_str += tabulate(metrics_data_print, headers=['class', 'SI', 'confidence', 
                                                        'localization', 'extent',
                                                        'heading'], tablefmt='orgtbl')
    return metrics_str


def parse_nuscenes_data(nusc, pred_infos, interval=1):
    scenes_collector = defaultdict(list)
    for pred_info in pred_infos:
        sample_token = pred_info['metadata']['token']
        gt_sample = nusc.get('sample', sample_token)
        scene_token = gt_sample['scene_token']
        scenes_collector[scene_token].append([gt_sample, pred_info])
    
    cur_det_annos, pre_det_annos, cur_gt_annos, pre_gt_annos = [], [], [], []
    for scene_token, info_list in tqdm(list(scenes_collector.items())):
        info_list = sorted(info_list, key=lambda x: x[0]['timestamp'])
        scene_gt_samples, scene_pred_infos = list(zip(*info_list))

        scene_gt_infos = []
        for sample in scene_gt_samples:
            sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

            boxes, instance_tokens, num_lidar_pts, gt_names = [], [], [], []
            for anno_token in sample['anns']:
                record = nusc.get('sample_annotation', anno_token)
                gt_name = map_name_from_general_to_detection[record['category_name']]
                if gt_name == 'ignore' or record['num_lidar_pts'] < 1:
                    continue
                
                gt_names.append(gt_name)
                instance_tokens.append(record['instance_token'])
                num_lidar_pts.append(record['num_lidar_pts'])
                box = Box(record['translation'], record['size'], Quaternion(record['rotation']),
                          name=record['category_name'], token=record['token'])
                box.velocity = nusc.box_velocity(box.token)
                # Move box to ego vehicle coord system
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)
                # Move box to sensor coord system
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

                v = np.dot(box.orientation.rotation_matrix, np.array([1, 0, 0]))
                yaw = np.arctan2(v[[1]], v[[0]])
                boxes.append(np.concatenate([box.center, box.wlh, yaw, box.velocity[:2]]))
            
            boxes = np.stack(boxes, axis=0) if boxes else np.zeros((0, 9), dtype=np.float32)
            instance_tokens = np.array(instance_tokens, dtype=np.dtype('<U32')) \
                if instance_tokens else np.zeros((0, ), dtype=np.dtype('<U32'))
            num_lidar_pts = np.array(num_lidar_pts, dtype=np.int) if num_lidar_pts \
                else np.zeros((0, ), dtype=np.int)
            gt_names = np.array(gt_names, dtype=np.dtype('<U20')) if gt_names \
                else np.zeros((0, ), dtype=np.dtype('<U20'))
            scene_gt_infos.append(dict(
                boxes=boxes, instance_tokens=instance_tokens, 
                num_lidar_pts=num_lidar_pts, gt_names=gt_names,
                token=sample['token'], timestamp=sample['timestamp']))

        cur_det_annos.extend(scene_pred_infos[interval:])
        cur_gt_annos.extend(scene_gt_infos[interval:])
        pre_det_annos.extend(scene_pred_infos[:-interval])
        pre_gt_annos.extend(scene_gt_infos[:-interval])
    
    assert len(cur_det_annos) == len(pre_det_annos) == len(cur_gt_annos) == len(pre_gt_annos)
    return cur_det_annos, pre_det_annos, cur_gt_annos, pre_gt_annos


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--data_root', type=str, default=None, help='nuscenes data path')
    parser.add_argument('--version', type=str, default=None, help='nuscenes version')
    parser.add_argument('--class_names', type=str, nargs='+', 
                        default=['car','truck', 'construction_vehicle', 'bus',
                                 'trailer', 'barrier', 'motorcycle', 'bicycle',
                                 'pedestrian', 'traffic_cone'])
    parser.add_argument('--interval', type=int, default=2, help='sampled interval for GT sequences')
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.data_root)
    pred_infos = pickle.load(open(args.pred_infos, 'rb'))

    parsed_outputs = parse_nuscenes_data(nusc, pred_infos, interval=args.interval)
    si_dict = eval_nuscenes_stability_index(*parsed_outputs, class_names=args.class_names)
    si_str = print_stability_index_results(si_dict, args.class_names)
    print(si_str)

    # save to file
    log_file = '/'.join(args.pred_infos.split('/')[:-1]) + f'/eval_si_%s.log' %datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    print('Save log file to %s' % log_file)
    with open(log_file, 'w') as f:
        f.write(si_str)

if __name__ == '__main__':
    main()
