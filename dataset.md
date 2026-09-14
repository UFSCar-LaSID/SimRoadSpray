
# SimRoadSpray dataset documentation

## Dataset strucutre

## Scenes configurations

## `calib` folder

## `cam_images` folder

## `labels` folder

## `scenes_filtered` and `scenes_full` folders

This repository contains the binary labeled point clouds collected from the indoor spray experiments. The dataset is organized in the structure bellow:

```
<scene_folder>/
├── points/
│   ├── <xyz_intensity_pcd_1>.bin
│   ├── <xyz_intensity_pcd_2>.bin
│   └── ...
├── full_labels/
│   ├── <full_labels_1>.bin
│   ├── <full_labels_2>.bin
│   └── ...
└── spray_filter_labels/
    ├── <spray_filter_labels_1>.bin
    ├── <spray_filter_labels_2>.bin
    └── ...
```

Where each `scene_folder` contains the labeled point clouds of a specific scene. Each scene represents an experiment, with sequential frames (like a video) collected from it.

Each `scene_folder` contain **3** subfolders. These folders contain the binary files, each one being named with the timestamp of the data collection. Next, the **3** folders are explained in details:

1. The `points` folder: contains the original point clouds in the binary (`.bin`) format. These point clouds have the xyz coordinates and the intesity of each point in the point cloud (totalizing 4 columns). It is possible to read this file using `numpy`, as shown in the code snipet:

    ```
    np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    ```

2. The `full_labels` folder: contains the original labels in the binary (`.bin`) format. It is one array containing the labeled point cloud using the original classes (background, target, car and spray). These labels can be used for data analysis. It is possible to read this file using `numpy`, as shown in the code snipet:

    ```
    np.fromfile(file_path, dtype=np.uint8)
    ```

3. The `spray_filter_labels` folder: contains the spray filter labels in the binary (`.bin`) format. It is one array containing the labeled point cloud using the not spray class (1 - car, target, background) and spray (0 - spray). This can be used to train models to learn how to filter out the water spray. It is possible to read this file using `numpy`, as shown in the code snipet:

    ```
    np.fromfile(file_path, dtype=np.uint8)
    ```