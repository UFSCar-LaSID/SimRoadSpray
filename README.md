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

<table>
  <thead>
    <tr>
      <th>Model name</th>
      <th>Model weights</th>
      <th>Download command</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>BEVFusion-LiDAR</td>
      <td><a href="https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth">Download</a></td>
      <td>
        wget https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth -O bevfusion_lidar.pth
      </td>
    </tr>
    <tr>
      <td>CenterPoint</td>
      <td><a href="https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth">Download</a></td>
      <td>
        wget https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth -O centerpoint.pth
      </td>
    </tr>
    <tr>
      <td>SSN</td>
      <td><a href="https://download.openmmlab.com/mmdetection3d/v1.0.0_models/ssn/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d_20210829_210615-361e5e04.pth">Download</a></td>
      <td>
        wget https://download.openmmlab.com/mmdetection3d/v1.0.0_models/ssn/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d/hv_ssn_regnet-400mf_secfpn_sbn-all_2x16_2x_nus-3d_20210829_210615-361e5e04.pth -O ssn.pth
      </td>
    </tr>
    <tr>
      <td>PointPillars</td>
      <td><a href="https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth">Download</a></td>
      <td>
        wget https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth -O pointpillars.pth
      </td>
    </tr>
  </tbody>
</table>

### 3. Packages Installation

```
docker build -t SimRoadSpray .
```

```
```

### 4. Run models evaluation

### 5. Generate plots and tables

## Citation