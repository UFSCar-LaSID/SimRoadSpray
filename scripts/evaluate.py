
import numpy as np
import torch
import plotly.express as px

from mmdet3d.evaluation.functional.indoor_eval import average_precision


# Adapted from https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/evaluation/functional/indoor_eval.py#L56-L161
def calc_metrics(pred, gt, iou_thr=None, confidence_thr=0.05):
    """Generic functions to compute precision/recall for object detection for a
    single class.

    Args:
        pred (dict): Predictions mapping from image id to bounding boxes
            and scores.
        gt (dict): Ground truths mapping from image id to bounding boxes.
        iou_thr (list[float]): A list of iou thresholds.
        confidence_thr (float): The confidence threshold.

    Return:
        tuple (np.ndarray, np.ndarray, float): Recalls, precisions and
            average precision.
    """

    # {img_id: {'bbox': box structure, 'det': matched list}}
    class_recs = {}
    npos = 0
    for img_id in gt.keys():
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
    for img_id in pred.keys():
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
        ap = average_precision(recall, precision)[0]
        
        if confidence_thr is not None:
            # Filter out detections below the confidence threshold
            index = np.where(confidence[sorted_ind] >= confidence_thr)[0][-1]
            tp = tp[index]
            fp = fp[index]
            recall = recall[index]
            precision = precision[index]
        
        ret.append({
            'tp': tp,
            'fp': fp,
            'recall': recall,
            'precision': precision,
            'ap': ap
        })

    return ret

def save_preds_as_json(preds, save_path):
    import json
    
    preds_serializable = {}
    for key, value in preds.items():
        boxes, scores = value
        boxes_list = boxes.tensor.cpu().numpy().tolist()
        scores_list = scores.tolist()
        preds_serializable[key] = [boxes_list, scores_list]
    
    with open(save_path, 'w') as f:
        json.dump(preds_serializable, f)

if __name__ == '__main__':
    import argparse
    from mmdet3d.datasets.indoor_spray_dataset import IndoorSprayDataset
    from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
    import os
    from tqdm import tqdm
    from mmdet3d.apis import inference_detector
    from mmengine.config import Config
    from mmengine.runner import Runner
    import pandas as pd

    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to model file')
    parser.add_argument('--add_zeros_col', action='store_true', help='add zeros column to points')
    parser.add_argument('--save_path', type=str, help='Path to save results')
    parser.add_argument('--clamp_boxes', action='store_true', help='Clamp predicted boxes to the scene limits')
    parser.add_argument('--extra_z_value', type=float, default=0.0, help='Value to add to the z-coordinate of the points and GT boxes (useful for adjusting for sensor height differences)')
    parser.add_argument('--z_min', type=float, default=None, help='Minimum z values for points in the point cloud (if not provided, no clamping will be applied)')
    args = parser.parse_args()
    
    cfg_path = args.config
    checkpoint_path = args.checkpoint
    add_zeros_col = args.add_zeros_col
    save_path = args.save_path
    clamp_boxes = args.clamp_boxes
    extra_z_value = args.extra_z_value
    z_min = args.z_min
    cfg = Config.fromfile(cfg_path)
    cfg.work_dir = os.path.join('/mmdetection3d/work_dirs')
    #cfg.log_level = 'ERROR'  # Stop printing
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(checkpoint_path)
    model = runner.model
    model.cfg = cfg
    model.eval()
    
    dry_scenes_names = [
        '10_scenario-a_dry_10m',
        '12_scenario-a_dry_20m',
        '14_scenario-a_dry_30m',
        
        '16_scenario-b_dry_10m',
        '18_scenario-b_dry_20m',
        '20_scenario-b_dry_30m',
        
        '22_scenario-c_dry_5m',
        '24_scenario-c_dry_10m'
    ]
    
    spray_scenes_names = [
        '11_scenario-a_spray-2_10m',
        '13_scenario-a_spray-2_20m',
        '15_scenario-a_spray-2_30m',
        
        '17_scenario-b_spray-2_10m',
        '19_scenario-b_spray-2_20m',
        '21_scenario-b_spray-2_30m',
        
        '23_scenario-c_spray-2_5m',
        '25_scenario-c_spray-2_10m'
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
    
    ious = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    ious_keys = {}  # Initialize ious_keys with all IoU thresholds for both dry and spray scenes (this will help in the sorting)
    for iou in ious:
        ious_keys[f'AP_{str(iou).replace(".", "")}_dry'] = 0
        ious_keys[f'AP_{str(iou).replace(".", "")}_spray'] = 0
    
    df_info = [
        { 'scene': 'A 10m', **ious_keys },
        { 'scene': 'A 20m', **ious_keys },
        { 'scene': 'A 30m', **ious_keys },
        { 'scene': 'B 10m', **ious_keys },
        { 'scene': 'B 20m', **ious_keys },
        { 'scene': 'B 30m', **ious_keys },
        { 'scene': 'C 5m',  **ious_keys },
        { 'scene': 'C 10m', **ious_keys }
    ]
    
    os.makedirs(save_path, exist_ok=True)
    
    def generate_preds(scene_idx, scene_name, condition):
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
        
        preds = {}
        gts = {}
        
        for i in range(len(dataset)):
            points = dataset[i]['inputs']['points'].numpy()
            if add_zeros_col:
                zeros_col = np.zeros((points.shape[0], 1), dtype=points.dtype)
                points = np.hstack((points, zeros_col))
            points[:, 2] += extra_z_value
            if z_min is not None:
                points = points[points[:, 2] >= z_min]

            result, _ = inference_detector(model, points)
            
            boxes_3d = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
            if clamp_boxes:
                boxes_3d[:, 0] += (boxes_3d[:, 3] - np.clip(boxes_3d[:, 3], a_min=0.01, a_max=6.20)) / 2
                #boxes_3d[:, 1] += (boxes_3d[:, 4] - np.clip(boxes_3d[:, 4], a_min=0.01, a_max=2.10)) / 2
                boxes_3d[:, 2] += (boxes_3d[:, 5] - np.clip(boxes_3d[:, 5], a_min=0.01, a_max=2.50)) / 2
                boxes_3d[:, 3] = np.clip(boxes_3d[:, 3], a_min=0.01, a_max=6.20)
                boxes_3d[:, 4] = np.clip(boxes_3d[:, 4], a_min=0.01, a_max=2.30)
                boxes_3d[:, 5] = np.clip(boxes_3d[:, 5], a_min=0.01, a_max=2.50)
                boxes_3d[:, 6] = 0
            cur_pred_boxes = LiDARInstance3DBoxes(boxes_3d[:, :7]).to('cpu')  # nuScenes generates vx and vy, but we only need the first 7 dimensions for evaluation
            cur_pred_scores = result.pred_instances_3d.scores_3d.cpu().numpy()
            cur_preds = [cur_pred_boxes, cur_pred_scores]
            cur_gt = dataset[i]['data_samples'].gt_instances_3d.bboxes_3d
            cur_gt.tensor[:, 2] += extra_z_value  # Adjust the z-coordinate of the ground truth boxes to account for the sensor height difference

            
            preds[dataset[i]['data_samples'].sample_idx] = cur_preds
            gts[dataset[i]['data_samples'].sample_idx] = cur_gt
        
        metrics = calc_metrics(preds, gts, iou_thr=ious)
        
        for i, iou in enumerate(ious):
            df_info[scene_idx][f'AP_{str(iou).replace(".", "")}_{condition}'] = metrics[i]['ap']
            df_info[scene_idx][f'TP_{str(iou).replace(".", "")}_{condition}'] = metrics[i]['tp']
            df_info[scene_idx][f'FP_{str(iou).replace(".", "")}_{condition}'] = metrics[i]['fp']
        
        preds_save_path = os.path.join(save_path, scene_name, 'preds.json')
        os.makedirs(os.path.dirname(preds_save_path), exist_ok=True)
        
        save_preds_as_json(preds, preds_save_path)

    for scene_idx, scene_name in tqdm(enumerate(dry_scenes_names), desc='Processing dry scenes', total=len(dry_scenes_names)):
        generate_preds(scene_idx, scene_name, 'dry')
        
    for scene_idx, scene_name in tqdm(enumerate(spray_scenes_names), desc='Processing spray scenes', total=len(spray_scenes_names)):
        generate_preds(scene_idx, scene_name, 'spray')
    
    
    df = pd.DataFrame(df_info)
    
    mean_row = df.mean(numeric_only=True)
    median_row = df.median(numeric_only=True)
    df.loc[len(df)] = ['mean', *mean_row.values]
    df.loc[len(df)] = ['median', *median_row.values]

    df.to_csv(os.path.join(save_path, 'evaluation_results.csv'), index=False)
    df.to_markdown(os.path.join(save_path, 'evaluation_results.md'), index=False)
    
    raw_data = []

    for th in ious:
        for index, row in df.iterrows():
            th_name = str(th).replace(".", "")
            scene_name = row['scene'].replace(' ', '_')
            raw_data.append(
                {
                    'threshold': th,
                    'AP': row[f'AP_{str(th).replace(".", "")}_spray'],
                    'color': f'{scene_name}_spray'
                }
            )
            raw_data.append(
                {
                    'threshold': th,
                    'AP': row[f'AP_{str(th).replace(".", "")}_dry'],
                    'color': f'{scene_name}_dry'
                }
            )
    df_plot = pd.DataFrame(raw_data)
    fig = px.line(df_plot, x='threshold', y='AP', color='color')

    fig.update_layout(
        title='AP vs Threshold for Different Scenes and Conditions',
        xaxis_title='IoU Threshold',
        yaxis_title='Average Precision (AP)',
        legend_title='Scene and Condition',
        template='plotly_white'
    )
    # save as html
    fig.write_html(os.path.join(save_path, 'ap_vs_threshold_plot.html'))
