
from mmengine.dataset import BaseDataset
from mmdet3d.registry import DATASETS
import os
import numpy as np
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes


# Adapted code from https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/kitti_data_utils.py#L117-L158
# and https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/update_infos_to_v2.py#L459-L511
def get_bboxes(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    content = [line.strip().split(' ') for line in lines]
    dims = np.array([[float(info) for info in x[8:11]]
                      for x in content]).reshape(-1, 3)[:, [1, 0, 2]]
    loc = np.array([[float(info) for info in x[11:14]]
                     for x in content]).reshape(-1, 3)
    loc[:, 2] -= dims[:, 2] / 2
    ry = np.array([float(x[14])for x in content]).reshape(-1)
    
    gt_bboxes_3d = np.concatenate([loc, dims, ry[:, None]], axis=1)
    
    return LiDARInstance3DBoxes(gt_bboxes_3d)

@DATASETS.register_module()
class SimRoadSprayDataset(BaseDataset):

    
    def __init__(self,
        scenes_to_use=None,
        scenes_to_not_use=None, 
        split_size=1,
        **kwargs
    ) -> None:
        '''
        Args:
            scenes_to_use: Which scenes to use. If None, use all. Can pass a dictionary with filter parameters, e.g. {'scenario': ['target'], 'distance': [10, 20]} to only use scenes matching those parameters
            scenes_to_not_use: Which scenes to not use. If None, remove none. Can pass a dictionary with filter parameters, e.g. {'scenario': ['target'], 'distance': [10, 20]} to not use scenes matching those parameters
            split_size: value between 0 and 1 indicating the fraction of data to use (e.g. 0.5 means use only the first half of the data)
        '''
        self.scenes_to_use = scenes_to_use
        self.scenes_to_not_use = scenes_to_not_use
        self.split_size = split_size
        self.scenes_info = [
            {
                'scene_number': 1,
                'scene_folder': '01_target_dry_10m',
                'spray': False,
                'distance': 10,
                'scenario': 'target',
                'intensity': 0
            },
            {
                'scene_number': 2,
                'scene_folder': '02_target_spray-1_10m',
                'spray': True,
                'distance': 10,
                'scenario': 'target',
                'intensity': 1
            },
            {
                'scene_number': 3,
                'scene_folder': '02_target_spray-2_10m',
                'spray': True,
                'distance': 10,
                'scenario': 'target',
                'intensity': 2
            },
            
            {
                'scene_number': 4,
                'scene_folder': '04_target_dry_20m',
                'spray': False,
                'distance': 20,
                'scenario': 'target',
                'intensity': 0
            },
            {
                'scene_number': 5,
                'scene_folder': '05_target_spray-1_20m',
                'spray': True,
                'distance': 20,
                'scenario': 'target',
                'intensity': 1
            },
            {
                'scene_number': 6,
                'scene_folder': '06_target_spray-2_20m',
                'spray': True,
                'distance': 20,
                'scenario': 'target',
                'intensity': 2
            },
            
            {
                'scene_number': 7,
                'scene_folder': '07_target_dry_30m',
                'spray': False,
                'distance': 30,
                'scenario': 'target',
                'intensity': 0
            },
            {
                'scene_number': 8,
                'scene_folder': '08_target_spray-1_30m',
                'spray': True,
                'distance': 30,
                'scenario': 'target',
                'intensity': 1
            },
            {
                'scene_number': 9,
                'scene_folder': '09_target_spray-2_30m',
                'spray': True,
                'distance': 30,
                'scenario': 'target',
                'intensity': 2
            },
            
            {
                'scene_number': 10,
                'scene_folder': '10_scenario-a_dry_10m',
                'spray': False,
                'distance': 10,
                'scenario': 'scenario_a',
                'intensity': 0
            },
            {
                'scene_number': 11,
                'scene_folder': '11_scenario-a_spray-2_10m',
                'spray': True,
                'distance': 10,
                'scenario': 'scenario_a',
                'intensity': 2
            },
            
            {
                'scene_number': 12,
                'scene_folder': '12_scenario-a_dry_20m',
                'spray': False,
                'distance': 20,
                'scenario': 'scenario_a',
                'intensity': 0
            },
            {
                'scene_number': 13,
                'scene_folder': '13_scenario-a_spray-2_20m',
                'spray': True,
                'distance': 20,
                'scenario': 'scenario_a',
                'intensity': 2
            },
            
            {
                'scene_number': 14,
                'scene_folder': '14_scenario-a_dry_30m',
                'spray': False,
                'distance': 30,
                'scenario': 'scenario_a',
                'intensity': 0
            },
            {
                'scene_number': 15,
                'scene_folder': '15_scenario-a_spray-2_30m',
                'spray': True,
                'distance': 30,
                'scenario': 'scenario_a',
                'intensity': 2
            },
            
            {
                'scene_number': 16,
                'scene_folder': '16_scenario-b_dry_10m',
                'spray': False,
                'distance': 10,
                'scenario': 'scenario_b',
                'intensity': 0
            },
            {
                'scene_number': 17,
                'scene_folder': '17_scenario-b_spray-2_10m',
                'spray': True,
                'distance': 10,
                'scenario': 'scenario_b',
                'intensity': 2
            },
            
            {
                'scene_number': 18,
                'scene_folder': '18_scenario-b_dry_20m',
                'spray': False,
                'distance': 20,
                'scenario': 'scenario_b',
                'intensity': 0
            },
            {
                'scene_number': 19,
                'scene_folder': '19_scenario-b_spray-2_20m',
                'spray': True,
                'distance': 20,
                'scenario': 'scenario_b',
                'intensity': 2
            },
            
            {
                'scene_number': 20,
                'scene_folder': '20_scenario-b_dry_30m',
                'spray': False,
                'distance': 30,
                'scenario': 'scenario_b',
                'intensity': 0
            },
            {
                'scene_number': 21,
                'scene_folder': '21_scenario-b_spray-2_30m',
                'spray': True,
                'distance': 30,
                'scenario': 'scenario_b',
                'intensity': 2
            },
            
            {
                'scene_number': 22,
                'scene_folder': '22_scenario-c_dry_5m',
                'spray': False,
                'distance': 5,
                'scenario': 'scenario_c',
                'intensity': 0
            },
            {
                'scene_number': 23,
                'scene_folder': '23_scenario-c_spray-2_5m',
                'spray': True,
                'distance': 5,
                'scenario': 'scenario_c',
                'intensity': 2
            },
            
            {
                'scene_number': 24,
                'scene_folder': '24_scenario-c_dry_10m',
                'spray': False,
                'distance': 10,
                'scenario': 'scenario_c',
                'intensity': 0
            },
            {
                'scene_number': 25,
                'scene_folder': '25_scenario-c_spray-2_10m',
                'spray': True,
                'distance': 10,
                'scenario': 'scenario_c',
                'intensity': 2
            },
        ]

        self.selected_scenes = []
        if self.scenes_to_use is None: 
            self.selected_scenes = self.scenes_info
        else:
            for scene in self.scenes_info:
                add_scene = True
                for filter_key, filter_values in self.scenes_to_use.items():
                    if scene[filter_key] not in filter_values:
                        add_scene = False
                        break
                if add_scene:
                    self.selected_scenes.append(scene)
        
        if self.scenes_to_not_use is not None:
            selected_scenes_copy = self.selected_scenes.copy()
            for scene in selected_scenes_copy:
                remove_scene = False
                for filter_key, filter_values in self.scenes_to_not_use.items():
                    if scene[filter_key] in filter_values:
                        remove_scene = True
                        break
                if remove_scene:
                    self.selected_scenes.remove(scene)
        
        super().__init__(**kwargs)
    
    def _join_prefix(self):
        pass

    def load_data_list(self):
        data_list = []
        for scene in self.selected_scenes:
            scene_folder = scene['scene_folder']
            timestamps = [filename.replace('.bin', '') for filename in  os.listdir(os.path.join(self.data_root, scene_folder, self.data_prefix['pts']))]
            split_idx = int(len(timestamps) * self.split_size)
            if self.test_mode:
                timestamps = timestamps[-split_idx:]
            else:
                timestamps = timestamps[:split_idx]

            for i, timestamp in enumerate(timestamps):
                gt_bboxes_3d = get_bboxes(os.path.join(self.data_root, '..', 'labels', scene_folder, f'{timestamp}.txt'))
                labels_3d = np.zeros(gt_bboxes_3d.shape[0], dtype=int)  # all points belong to the same class (car)
                data_info = {
                    'scene_name': scene_folder,
                    'timestamp': timestamp,
                    'sample_idx': len(data_list),
                    'idx_within_scene': i,
                    'lidar_points': {'lidar_path': os.path.join(self.data_root, scene_folder, self.data_prefix['pts'], f'{timestamp}.bin')},
                    'pts_semantic_mask_path': os.path.join(self.data_root, scene_folder, self.data_prefix['pts_semantic_mask'], f'{timestamp}.bin'),
                    'ann_info': {
                        'gt_bboxes_3d': gt_bboxes_3d,
                        'gt_labels_3d': labels_3d
                    }
                }
                data_list.append(data_info)
        return data_list