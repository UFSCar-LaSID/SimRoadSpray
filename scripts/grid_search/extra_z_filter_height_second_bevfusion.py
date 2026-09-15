
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
parser.add_argument('--extra_height_range', type=float, nargs=3, required=True)
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

extra_z_values = np.arange(*args.extra_z_range)
extra_height_values = np.arange(*args.extra_height_range)

pbar = tqdm(total=len(extra_z_values) * len(extra_height_values), desc='Grid Search for extra_z')

for extra_z in extra_z_values:
    for extra_height in extra_height_values:
        grid_search.add_test(
            f'z={extra_z:.4f}_h={extra_height:.4f}', 
            model, 
            transform_points=None,
            clamp_boxes=True,
            extra_z_value=extra_z,
            add_zeros_col=True,
            z_min=(-1.5+extra_z+extra_height)
        )
        
        pbar.update(1)

os.makedirs(root_save_path, exist_ok=True)
grid_search.print_and_save_results(os.path.join(root_save_path, 'extra_z_search.csv'))