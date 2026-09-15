
from copy import deepcopy

import pandas as pd
from mmdet3d.datasets.indoor_spray_dataset import IndoorSprayDataset
from mmdet3d.apis import inference_detector
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
import torch
import numpy as np
from mmdet3d.evaluation.functional.indoor_eval import average_precision

# Adapted from https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/evaluation/functional/indoor_eval.py#L56-L161
def calc_metrics(pred, gt, iou_thr=None):
    """Generic functions to compute precision/recall for object detection for a
    single class.

    Args:
        pred (dict): Predictions mapping from image id to bounding boxes
            and scores.
        gt (dict): Ground truths mapping from image id to bounding boxes.
        iou_thr (list[float]): A list of iou thresholds.

    Return:
        tuple (np.ndarray, np.ndarray, float): Recalls, precisions and
            average precision.
    """

    # {img_id: {'bbox': box structure, 'det': matched list}}
    class_recs = {}
    npos = 0
    for img_id in range(len(gt)):
        cur_gt_num = len(gt[img_id])
        if cur_gt_num != 0:
            gt_cur = torch.zeros([cur_gt_num, 7], dtype=torch.float32)
            for i in range(cur_gt_num):
                gt_cur[i] = gt[img_id][i].tensor
            bbox = gt[img_id][0].new_box(gt_cur)
        else:
            bbox = gt[img_id]
        det = [[False] * len(bbox) for i in iou_thr]
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}

    # construct dets
    image_ids = []
    confidence = []
    ious = []
    for img_id in range(len(pred)):
        cur_num = len(pred[img_id][0])
        if cur_num == 0:
            continue
        pred_cur = torch.zeros((cur_num, 7), dtype=torch.float32)
        box_idx = 0

        for i in range(cur_num):
            box = pred[img_id][0][i]
            score = pred[img_id][1][i]
            image_ids.append(img_id)
            confidence.append(score)
            pred_cur[box_idx] = box.tensor
            box_idx += 1
        pred_cur = box.new_box(pred_cur)
        gt_cur = class_recs[img_id]['bbox']
        if len(gt_cur) > 0:
            # calculate iou in each image
            iou_cur = pred_cur.overlaps(pred_cur, gt_cur)
            for i in range(cur_num):
                ious.append(iou_cur[i])
        else:
            for i in range(cur_num):
                ious.append(np.zeros(1))

    confidence = np.array(confidence)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    image_ids = [image_ids[x] for x in sorted_ind]
    ious = [ious[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp_thr = [np.zeros(nd) for i in iou_thr]
    fp_thr = [np.zeros(nd) for i in iou_thr]
    for d in range(nd):
        R = class_recs[image_ids[d]]
        iou_max = -np.inf
        BBGT = R['bbox']
        cur_iou = ious[d]

        if len(BBGT) > 0:
            # compute overlaps
            for j in range(len(BBGT)):
                # iou = get_iou_main(get_iou_func, (bb, BBGT[j,...]))
                iou = cur_iou[j]
                if iou > iou_max:
                    iou_max = iou
                    jmax = j

        for iou_idx, thresh in enumerate(iou_thr):
            if iou_max > thresh:
                if not R['det'][iou_idx][jmax]:
                    tp_thr[iou_idx][d] = 1.
                    R['det'][iou_idx][jmax] = 1
                else:
                    fp_thr[iou_idx][d] = 1.
            else:
                fp_thr[iou_idx][d] = 1.

    ret = []
    for iou_idx, thresh in enumerate(iou_thr):
        # compute precision recall
        fp = np.cumsum(fp_thr[iou_idx])
        tp = np.cumsum(tp_thr[iou_idx])
        recall = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = average_precision(recall, precision)
        ret.append((recall, precision, ap))

    return ret



class GridSearch:
    
    def __init__(self):
        scenes_names = [
            '11_scenario-a_spray-2_10m',
            '13_scenario-a_spray-2_20m',
            '15_scenario-a_spray-2_30m',
            
            '17_scenario-b_spray-2_10m',
            '19_scenario-b_spray-2_20m',
            '21_scenario-b_spray-2_30m',
            
            '23_scenario-c_spray-2_5m',
            '25_scenario-c_spray-2_10m',
            
            '10_scenario-a_dry_10m',
            '12_scenario-a_dry_20m',
            '14_scenario-a_dry_30m',
            
            '16_scenario-b_dry_10m',
            '18_scenario-b_dry_20m',
            '20_scenario-b_dry_30m',
            
            '22_scenario-c_dry_5m',
            '24_scenario-c_dry_10m'
        ]
        
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                backend_args=None),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_seg_3d=True,
                seg_3d_dtype='np.uint8'),
            dict(
                type='Pack3DDetInputs',
                keys=[
                    'points', 'pts_semantic_mask', 'gt_bboxes_3d', 'gt_labels_3d'
                ],
                meta_keys=[
                    'scene_name', 'sample_idx', 'idx_within_scene', 'timestamp'
                ])
        ]

        data_prefix = {
            'pts': 'points',
            'pts_semantic_mask': 'full_labels',
        }
        data_root = '/mmdetection3d/data/indoor_spray/scenes_filtered'
        metainfo = {
            'classes': [
                'background', 'target', 'car', 'spray'
            ]
        }
        
        self.all_points = []
        self.all_gts = []
        self.scenes_names = scenes_names
        
        for scene_name in scenes_names:
            dataset = IndoorSprayDataset(
                data_root=data_root,
                data_prefix=data_prefix,
                metainfo=metainfo,
                test_mode=False,
                scenes_to_use={
                    'scene_folder': [scene_name]
                },
                split_size=1,
                pipeline=pipeline
            )
            points = dataset[0]['inputs']['points'].numpy()
            cur_gt = dataset[0]['data_samples'].gt_instances_3d.bboxes_3d
            
            self.all_points.append(points)
            self.all_gts.append(cur_gt)
        
        self.results = []
    
    
    def add_test(self, name, model, transform_points, clamp_boxes=False, extra_z_value=0, add_zeros_col=False, z_min=None):
        try:
            preds = []
            for points, scene_name in zip(self.all_points, self.scenes_names):
                points = points.copy()
                if transform_points is not None:
                    points = transform_points(points)
                points[:, 2] += extra_z_value
                if add_zeros_col:
                    zeros_col = np.zeros((points.shape[0], 1), dtype=points.dtype)
                    points = np.hstack((points, zeros_col))
                if z_min is not None:
                    points = points[points[:, 2] >= z_min]
                
                result, _ = inference_detector(model, points)
                boxes_3d = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
                if clamp_boxes:
                    if 'scenario-c' in scene_name:
                        boxes_3d[:, 0] += (boxes_3d[:, 3] - np.clip(boxes_3d[:, 3], a_min=0.01, a_max=6.60)) / 2
                    else:
                        boxes_3d[:, 0] += (boxes_3d[:, 3] - np.clip(boxes_3d[:, 3], a_min=0.01, a_max=5.30)) / 2
                    #boxes_3d[:, 1] += (boxes_3d[:, 4] - np.clip(boxes_3d[:, 4], a_min=0.01, a_max=2.10)) / 2
                    #boxes_3d[:, 2] += (boxes_3d[:, 5] - np.clip(boxes_3d[:, 5], a_min=0.01, a_max=2.50)) / 2
                    if 'scenario-c' in scene_name:
                        boxes_3d[:, 3] = np.clip(boxes_3d[:, 3], a_min=0.01, a_max=6.60)
                    else:
                        boxes_3d[:, 3] = np.clip(boxes_3d[:, 3], a_min=0.01, a_max=5.30)
                    boxes_3d[:, 4] = np.clip(boxes_3d[:, 4], a_min=0.01, a_max=2.10)
                    boxes_3d[:, 5] = np.clip(boxes_3d[:, 5], a_min=0.01, a_max=2.50)
                    boxes_3d[:, 6] = 0
                cur_pred_boxes = LiDARInstance3DBoxes(boxes_3d[:, :7]).to('cpu')  # nuScenes generates vx and vy, but we only need the first 7 dimensions for evaluation
                cur_pred_scores = result.pred_instances_3d.scores_3d.cpu().numpy()
                cur_preds = [cur_pred_boxes, cur_pred_scores]
                preds.append(cur_preds)
            
            all_gts = deepcopy(self.all_gts)
            for i in range(len(all_gts)):
                all_gts[i].tensor[:, 2] += extra_z_value
                
            metrics = calc_metrics(preds, all_gts, iou_thr=[0.05, 0.1, 0.3, 0.5, 0.7])
            self.results.append(
                {
                    'name': name,
                    'AP@0.7': metrics[4][2][0],
                    'AP@0.5': metrics[3][2][0],
                    'AP@0.3': metrics[2][2][0],
                    'AP@0.1': metrics[1][2][0],
                    'AP@0.05': metrics[0][2][0]
                }
            )
        except Exception as e:
            self.results.append(
                {
                    'name': name,
                    'AP@0.7': -1,
                    'AP@0.5': -1,
                    'AP@0.3': -1,
                    'AP@0.1': -1,
                    'AP@0.05': -1
                }
            )
    
    def print_and_save_results(self, save_path):
        df = pd.DataFrame(self.results)
        df = df.sort_values(by=['AP@0.7', 'AP@0.5', 'AP@0.3', 'AP@0.1', 'AP@0.05'], ascending=False)
        print(df.head(20))
        df.to_csv(save_path, index=False)