
from mmengine.config import Config
from mmengine.runner import Runner
from grid_search import GridSearch
import os
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Grid Search for extra_z')
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--extra_z_range', type=float, nargs=3, required=True)
args = parser.parse_args()

root_save_path = args.save_path

cfg = Config.fromfile(args.config_path)
cfg.work_dir = os.path.join('/mmdetection3d/work_dirs')
cfg.log_level = 'ERROR'  # Stop printing
runner = Runner.from_cfg(cfg)
runner.load_checkpoint(args.checkpoint_path)
model = runner.model
model.cfg = cfg
model.eval()

grid_search = GridSearch()

for extra_z in tqdm(np.arange(*args.extra_z_range)):
    
    if 'bevfusion' in args.config_path:
        transform_points = lambda points: np.concatenate([points, np.zeros((points.shape[0], 1))], axis=1)
        new_point_cloud_range = [
            cfg.model.data_preprocessor.voxelize_cfg.point_cloud_range[0],
            cfg.model.data_preprocessor.voxelize_cfg.point_cloud_range[1],
            cfg.model.data_preprocessor.voxelize_cfg.point_cloud_range[2] + extra_z,
            cfg.model.data_preprocessor.voxelize_cfg.point_cloud_range[3],
            cfg.model.data_preprocessor.voxelize_cfg.point_cloud_range[4],
            cfg.model.data_preprocessor.voxelize_cfg.point_cloud_range[5] + extra_z
        ]
        model.data_preprocessor.voxelize_cfg = VoxelizationByGridShape(**dict(
            max_num_points=cfg.model.data_preprocessor.voxelize_cfg.max_num_points,
            point_cloud_range=new_point_cloud_range,
            voxel_size=cfg.model.data_preprocessor.voxelize_cfg.voxel_size,
            max_voxels=cfg.model.data_preprocessor.voxelize_cfg.max_voxels
        ))
    else:
        transform_points = None
        new_point_cloud_range = [
            cfg.model.data_preprocessor.voxel_layer.point_cloud_range[0],
            cfg.model.data_preprocessor.voxel_layer.point_cloud_range[1],
            cfg.model.data_preprocessor.voxel_layer.point_cloud_range[2] + extra_z,
            cfg.model.data_preprocessor.voxel_layer.point_cloud_range[3],
            cfg.model.data_preprocessor.voxel_layer.point_cloud_range[4],
            cfg.model.data_preprocessor.voxel_layer.point_cloud_range[5] + extra_z
        ]
        model.data_preprocessor.voxel_layer = VoxelizationByGridShape(**dict(
            max_num_points=cfg.model.data_preprocessor.voxel_layer.max_num_points,
            point_cloud_range=new_point_cloud_range,
            voxel_size=cfg.model.data_preprocessor.voxel_layer.voxel_size,
            max_voxels=cfg.model.data_preprocessor.voxel_layer.max_voxels
        ))

    grid_search.add_test(f'{extra_z:.6f}', model, transform_points=transform_points)

os.makedirs(root_save_path, exist_ok=True)
grid_search.print_and_save_results(os.path.join(root_save_path, 'extra_z_search.csv'))