# 4dgs-test-dataset
Synthetic Dataset to test the limits of 4DGS


4dgs-synthetic-dataset/
├── README.md
├── requirements.txt
├── setup.sh
│
├── scenarios/                         # MuJoCo XML files
│   ├── scene1_close_proximity.xml
│   ├── scene5_rapid_motion.xml
│   └── ...
│
├── trajectories/                      # Motion definitions
│   ├── __init__.py
│   ├── scene1_trajectories.py
│   ├── scene5_trajectories.py
│   └── trajectory_base.py
│
├── src/                               # Generation scripts
│   ├── viewer.py                     # python src/viewer.py --scene 1
│   ├── generate_dataset.py           # python src/generate_dataset.py --scene 1
│   ├── export_for_4dgs.py            # python src/export_for_4dgs.py --scene 1
│   ├── compute_bboxes.py
│   └── utils/
│       ├── mujoco_utils.py
│       ├── camera_utils.py
│       └── bbox_utils.py
│
├── configs/
│   ├── default_config.yaml
│   └── scene1_config.yaml
│
├── dataset/                           # Our organized format (for development)
│   └── scene1/
│       ├── images/
│       │   ├── cam0/
│       │   │   ├── frame_00000.png
│       │   │   └── ...
│       │   └── cam1/ ...
│       ├── bboxes/
│       ├── masks/
│       ├── camera_intrinsics.json
│       ├── camera_extrinsics.json
│       └── metadata.json
│
├── data/                              # 4DGS-compatible format (export target)
│   └── multipleview/
│       ├── scene1_close_proximity/
│       │   ├── cam00/
│       │   │   ├── frame_00001.jpg
│       │   │   ├── frame_00002.jpg
│       │   │   └── ...
│       │   ├── cam01/ ...
│       │   ├── cam07/
│       │   ├── sparse_/              # Run 4DGS's multipleviewprogress.sh
│       │   ├── points3D_multipleview.ply
│       │   └── poses_bounds_multipleview.npy
│       │
│       ├── scene5_rapid_motion/
│       └── ...
│
├── docs/
│   ├── dataset_format.md
│   ├── 4dgs_training.md              # How to train 4DGS on this dataset
│   └── scene_descriptions.md
│
└── examples/
    ├── train_4dgs.sh                 # Example training script
    └── evaluate_4dgs.py