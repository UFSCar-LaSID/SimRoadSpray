# SimRoadSpray

Official repository for "Beyond Rain: Investigating the Impact of Road Spray on LiDAR-Based Perception for ADAS".

## Abstract

Road spray is an underexplored adverse weather phenomenon that significantly degrades the performance of LiDAR-based perception systems in automated vehicles. Unlike rain or fog, spray is generated dynamically by moving vehicles on wet surfaces, making its systematic evaluation highly challenging due to uncontrollable field conditions and limited data availability. This paper proposes and evaluates a physical road spray testbed that enables reproducible road spray generation for controlled LiDAR sensor evaluation. Using a modular frame with individually configurable water nozzles, we create a semantically annotated point cloud dataset named SimRoadSpray, and benchmark four state-of-the-art LiDAR-based object detectors across three spatial scenarios and multiple distances. Experimental results show that, at the point cloud level, spray reduces the number of valid LiDAR returns by up to 33%, with point loss generally increasing with distance. This reduction in point density is more pronounced than the concurrent decrease in return intensity. Furthermore, spray adversely affects detection accuracy across all architectures, revealing significant architecture-dependent differences in robustness. Voxel-based detectors consistently outperform pillar-based methods, with BEVFusion LiDAR exhibiting the highest detection performance and the lowest median AP degradation (14.4%) under spray conditions.

## Results

## Dataset details

## Reproducing our results

This section describes the complete pipeline required to reproduce the experiments presented in this work. The process consists of five stages:

1. **Dataset download**: Download the SimRoadSpray dataset.
2. **Models download**: Download the models weights necessary to evaluate the 3D detections algorithms.
3. **Package installation**: Set up the required software environment by installing MMDetection3D and all project dependencies, either using Docker (recommended) or a local installation.
4. **3D detection models evaluation**: Evaluate the models under spray and dry conditions.
5. **Plots and tables generation**: Generate all figures and tables presented in the paper from the evaluation results using the provided Jupyter notebook.

### 1. Dataset download

The first step necessary to reproduce our results is to download the dataset, which includes de points clouds that will be analysed and processed by the 3D object detection models. The dataset is publicly available and can be downloaded in this link. [TODO LINK].

After downloading it is necessary to extract the files. You can extract wherever you want, but you need to remember the path to the dataset for the next steps. After extracting the files, it is expected to have the following folder structure:

```
<SimRoadSpray_path>/
├── calib/
├── cam_images/
├── labels/
├── scenes_filtered/
└── scenes_full/
```

### 2. Models download

It is necessary to download the models weights to be able to make the 3D detection inferences and evaluation. In the table bellow you can download the necessary models:

<table>
  <thead>
    <tr>
      <th>Model name</th>
      <th>Model weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>BEVFusion-LiDAR</td>
      <td>
        <a href="https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth">
          Download
        </a>
      </td>
    </tr>
    <tr>
      <td>CenterPoint</td>
      <td>
        <a href="https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth">
          Download
        </a>
      </td>
    </tr>
    <tr>
      <td>SSN</td>
      <td>
        <a href="https://download.openmmlab.com/mmdetection3d/v1.0.0_models/ssn/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d_20210829_210615-361e5e04.pth">
          Download
        </a>
      </td>
    </tr>
    <tr>
      <td>PointPillars</td>
      <td>
        <a href="https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth">
          Download
        </a>
      </td>
    </tr>
  </tbody>
</table>

Other option is to execute the command below inside the `/models` folder (if not exists, create it as an empty folder):

```
wget https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth -O bevfusion_lidar.pth
wget https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth -O centerpoint.pth
wget https://download.openmmlab.com/mmdetection3d/v1.0.0_models/ssn/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d_20210829_210615-361e5e04.pth -O ssn.pth
wget https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth -O pointpillars.pth
```

The models used in this project are trained on the [nuScenes dataset](https://www.nuscenes.org/nuscenes), and are publicly available by the [MMDetection3D library](https://github.com/open-mmlab/mmdetection3d).

### 3. Packages Installation

To run the experiments, you must install MMDetection3D along with the required Python dependencies. There are two supported installation methods:

1. Using Docker (recommended)
2. Installing from source (without Docker)

We strongly recommend using the Docker-based installation, as it provides a consistent environment and makes reproducing the experimental results significantly more reliable.

#### Docker installation (recommended)

Build the Docker image containing all the required dependencies by running:

```
docker build -t sim_road_spray .
```

Once the image has been built, create a Docker container with the following command:

```
docker run --gpus all --shm-size=8g -it -d \ 
  -v <SimRoadSpray_path>:/mmdetection3d/data/indoor_spray \ 
  -v ./configs:/mmdetection3d/extra_configs \ 
  -v ./models:/mmdetection3d/models \
  -v ./scripts:/mmdetection3d/scripts \ 
  sim_road_spray
```

Replace the placeholder `<nuscenes_path>` with where the SimRoadSpray dataset is (or will be) stored.

After creating the container, list the running containers to obtain its ID:

```
docker ps
```

Then attach to the container:

```
docker attach <container_id>
```

Once inside the container, you can proceed to the next steps.

#### Installation without docker

Alternatively, you can install MMDetection3D directly on your system. Follow the [official installation guide](https://mmdetection3d.readthedocs.io/en/latest/get_started.html) and install MMDetection3D from source.

### 4. Run models evaluation

With all installed, you can execute the following commands to generate the models predictions and evaluation.

#### BEVFusion-LiDAR

```
python /mmdetection3d/scripts/evaluate.py \
    --config /mmdetection3d/extra_configs/bevfusion_lidar.py \
    --checkpoint /mmdetection3d/models/bevfusion_lidar.pth \
    --save_path /mmdetection3d/results/bevfusion_lidar \
    --clamp_boxes --extra_z_value -0.4 --add_zeros_col --z_min -1.6
```

#### CenterPoint

```
python /mmdetection3d/scripts/evaluate.py \
    --config /mmdetection3d/extra_configs/centerpoint.py \
    --checkpoint /mmdetection3d/models/centerpoint.pth \
    --save_path /mmdetection3d/results/centerpoint \
    --clamp_boxes --extra_z_value -0.4 --add_zeros_col --z_min -1.6
```

#### SSN

```
python /mmdetection3d/scripts/evaluate.py \
    --config /mmdetection3d/extra_configs/ssn.py \
    --checkpoint /mmdetection3d/models/ssn.pth \
    --save_path /mmdetection3d/results/ssn \
    --clamp_boxes --extra_z_value -0.4 --add_zeros_col --z_min -1.6
```

#### PointPillars

```
python /mmdetection3d/scripts/evaluate.py \
    --config /mmdetection3d/extra_configs/pointpillars.py \
    --checkpoint /mmdetection3d/models/pointpillars.pth \
    --save_path /mmdetection3d/results/pointpillars \
    --clamp_boxes --extra_z_value -0.4 --add_zeros_col --z_min -1.6
```

### 5. Generate plots and tables

After completing the evaluation, generate the figures and tables used in the analysis by executing all cells in the [`plots_and_tables.ipynb`](/scripts/plots_and_tables.ipynb) Jupyter notebook.

Ensure that the evaluation has been completed successfully and that the results are available in the expected location before running the notebook. By default, the notebook reads the results from `/mmdetection3d/results`. If the evaluation results are stored in a different directory, update the corresponding path in the notebook before execution. The tables and plots will be saved in `/mmdetection3d/plots_and_tables`.

## Citation

```
```