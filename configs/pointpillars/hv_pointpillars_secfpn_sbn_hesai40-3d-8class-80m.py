_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_aicv-8class-voxel0.25-80m.py',
    '../_base_/datasets/hesai40-3d-8class-80m.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

model = dict(
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=8,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-80, -80, -1.70, 80, 80, -1.70],
                    [-80, -80, -1.52, 80, 80, -1.52],
                    [-80, -80, -1.67, 80, 80, -1.67],
                    [-80, -80, -1.73, 80, 80, -1.73],
                    [-80, -80, -1.45, 80, 80, -1.45],
                    [-80, -80, -1.70, 80, 80, -1.70],
                    [-80, -80, -1.78, 80, 80, -1.78],
                    [-80, -80, -1.44, 80, 80, -1.44]],
            sizes=[[3.45, 1.62, 1.34],      # Car
                   [6.24, 2.23, 2.35],      # Bus
                   [1.51, 0.61, 1.03],      # NonMot
                   [1.54, 0.81, 1.43],      # Cyclist
                   [0.51, 0.60, 1.30],      # Pedestrian
                   [0.30, 0.31, 0.61],      # TrafficCone
                   [0.89, 0.42, 0.59],      # Barrier
                   [1.02, 0.77, 0.76]],     # Others
            rotations=[0.7854, 2.3562],
            reshape_out=False)))

# runtime settings
lr_config = dict(step=[56, 59])
runner = dict(max_epochs=60)
evaluation = dict(interval=4)
