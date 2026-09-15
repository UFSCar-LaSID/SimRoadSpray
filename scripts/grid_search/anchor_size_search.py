
from mmengine.config import Config
from mmengine.runner import Runner
from grid_search import GridSearch
import os
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape
from tqdm import tqdm
import numpy as np
from mmdet3d.registry import MODELS
import argparse

parser = argparse.ArgumentParser(description='Grid Search for extra_z')
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--extra_x_size_range', type=float, nargs=3, required=True)
parser.add_argument('--extra_y_size_range', type=float, nargs=3, required=True)
parser.add_argument('--extra_z_size_range', type=float, nargs=3, required=True)
args = parser.parse_args()

root_save_path = args.save_path

cfg = Config.fromfile(args.config_path)
cfg.work_dir = os.path.join('/mmdetection3d/work_dirs')
cfg.log_level = 'ERROR'  # Stop printing
runner = Runner.from_cfg(cfg)

grid_search = GridSearch()

extra_x_size_values = np.arange(*args.extra_x_size_range)
extra_y_size_values = np.arange(*args.extra_y_size_range)
extra_z_size_values = np.arange(*args.extra_z_size_range)

pbar = tqdm(total=len(extra_x_size_values) * len(extra_y_size_values) * len(extra_z_size_values))

for extra_x_size in extra_x_size_values:
    for extra_y_size in extra_y_size_values:
        for extra_z_size in extra_z_size_values:
            cfg = Config.fromfile(args.config_path)
            for i, size in enumerate(cfg.model.bbox_head.anchor_generator.sizes):
                cfg.model.bbox_head.anchor_generator.sizes[i] = [
                    size[0] + extra_x_size,
                    size[1] + extra_y_size,
                    size[2] + extra_z_size
                ]
            cfg.work_dir = os.path.join('/mmdetection3d/work_dirs')
            cfg.log_level = 'ERROR'  # Stop printing
            runner = Runner.from_cfg(cfg)
            runner.load_checkpoint(args.checkpoint_path)
            model = runner.model
            model.cfg = cfg
            model.eval()

            grid_search.add_test(
                name=f'second_x{extra_x_size:.2f}_y{extra_y_size:.2f}_z{extra_z_size:.2f}',
                model=model,
                transform_points=None
            )
            pbar.update(1)

os.makedirs(root_save_path, exist_ok=True)
grid_search.print_and_save_results(os.path.join(root_save_path, 'results_anchor_size.csv'))