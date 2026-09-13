# SimRoadSpray

GitHub repository under construction

## Abstract

## Results

## Dataset details

## Reproducing our results

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
    --clamp_boxes --extra_z_value -0.2 --add_zeros_col
```

#### CenterPoint

```
python /mmdetection3d/scripts/evaluate.py \
    --config /mmdetection3d/extra_configs/centerpoint.py \
    --checkpoint /mmdetection3d/models/centerpoint.pth \
    --save_path /mmdetection3d/results/centerpoint \
    --clamp_boxes --extra_z_value -0.2 --add_zeros_col
```

#### SSN

```
python /mmdetection3d/scripts/evaluate.py \
    --config /mmdetection3d/extra_configs/ssn.py \
    --checkpoint /mmdetection3d/models/ssn.pth \
    --save_path /mmdetection3d/results/ssn \
    --clamp_boxes --extra_z_value -0.2 --add_zeros_col
```

#### PointPillars

```
python /mmdetection3d/scripts/evaluate.py \
    --config /mmdetection3d/extra_configs/pointpillars.py \
    --checkpoint /mmdetection3d/models/pointpillars.pth \
    --save_path /mmdetection3d/results/pointpillars \
    --clamp_boxes --extra_z_value -0.2 --add_zeros_col
```

### 5. Generate plots and tables

## Citation