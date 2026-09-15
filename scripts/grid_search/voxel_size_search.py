
from mmengine.config import Config
from mmengine.runner import Runner
from grid_search import GridSearch
import os
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape
from tqdm import tqdm
import numpy as np
from mmdet3d.registry import MODELS
import argparse
from math import ceil
from projects.BEVFusion.bevfusion.ops import Voxelization


parser = argparse.ArgumentParser(description='Grid Search for extra_z')
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--multiplier_x_size_range', type=float, nargs=3, required=True)
parser.add_argument('--multiplier_y_size_range', type=float, nargs=3, required=True)
parser.add_argument('--multiplier_z_size_range', type=float, nargs=3, required=True)
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

multiplier_x_size_values = np.arange(*args.multiplier_x_size_range)
multiplier_y_size_values = np.arange(*args.multiplier_y_size_range)
multiplier_z_size_values = np.arange(*args.multiplier_z_size_range)

pbar = tqdm(total=len(multiplier_x_size_values) * len(multiplier_y_size_values) * len(multiplier_z_size_values))

for multiplier_x_size in multiplier_x_size_values:
    for multiplier_y_size in multiplier_y_size_values:
        for multiplier_z_size in multiplier_z_size_values:
            if 'bevfusion' in args.config_path:
                transform_points = lambda points: np.concatenate([points, np.zeros((points.shape[0], 1))], axis=1)
                new_point_cloud_range = [
                    ceil(cfg.model.data_preprocessor.voxelize_cfg.point_cloud_range[0] * multiplier_x_size),
                    ceil(cfg.model.data_preprocessor.voxelize_cfg.point_cloud_range[1] * multiplier_y_size),
                    ceil(cfg.model.data_preprocessor.voxelize_cfg.point_cloud_range[2] * multiplier_z_size),
                    ceil(cfg.model.data_preprocessor.voxelize_cfg.point_cloud_range[3] * multiplier_x_size),
                    ceil(cfg.model.data_preprocessor.voxelize_cfg.point_cloud_range[4] * multiplier_y_size),
                    ceil(cfg.model.data_preprocessor.voxelize_cfg.point_cloud_range[5] * multiplier_z_size)
                ]
                new_voxel_size = [
                    cfg.model.data_preprocessor.voxelize_cfg.voxel_size[0] * multiplier_x_size,
                    cfg.model.data_preprocessor.voxelize_cfg.voxel_size[1] * multiplier_y_size,
                    cfg.model.data_preprocessor.voxelize_cfg.voxel_size[2] * multiplier_z_size
                ]
                model.pts_voxel_layer = Voxelization(**dict(
                    max_num_points=cfg.model.data_preprocessor.voxelize_cfg.max_num_points,
                    point_cloud_range=new_point_cloud_range,
                    voxel_size=new_voxel_size,
                    max_voxels=cfg.model.data_preprocessor.voxelize_cfg.max_voxels,
                    deterministic=True
                ))
            else:
                transform_points = None
                new_point_cloud_range = [
                    ceil(cfg.model.data_preprocessor.voxel_layer.point_cloud_range[0] * multiplier_x_size),
                    ceil(cfg.model.data_preprocessor.voxel_layer.point_cloud_range[1] * multiplier_y_size),
                    ceil(cfg.model.data_preprocessor.voxel_layer.point_cloud_range[2] * multiplier_z_size),
                    ceil(cfg.model.data_preprocessor.voxel_layer.point_cloud_range[3] * multiplier_x_size),
                    ceil(cfg.model.data_preprocessor.voxel_layer.point_cloud_range[4] * multiplier_y_size),
                    ceil(cfg.model.data_preprocessor.voxel_layer.point_cloud_range[5] * multiplier_z_size)
                ]
                new_voxel_size = [
                    cfg.model.data_preprocessor.voxel_layer.voxel_size[0] * multiplier_x_size,
                    cfg.model.data_preprocessor.voxel_layer.voxel_size[1] * multiplier_y_size,
                    cfg.model.data_preprocessor.voxel_layer.voxel_size[2] * multiplier_z_size
                ]
                model.data_preprocessor.voxel_layer = VoxelizationByGridShape(**dict(
                    max_num_points=cfg.model.data_preprocessor.voxel_layer.max_num_points,
                    point_cloud_range=new_point_cloud_range,
                    voxel_size=new_voxel_size,
                    max_voxels=cfg.model.data_preprocessor.voxel_layer.max_voxels
                ))

            grid_search.add_test(
                name=f'second_x{multiplier_x_size:.2f}_y{multiplier_y_size:.2f}_z{multiplier_z_size:.2f}',
                model=model,
                clamp_boxes=True,
                transform_points=None,
                extra_z_value=-0.2,
                add_zeros_col=True,
                z_min=-1.5
            )
            pbar.update(1)

os.makedirs(root_save_path, exist_ok=True)
grid_search.print_and_save_results(os.path.join(root_save_path, 'results_voxel_size.csv'))