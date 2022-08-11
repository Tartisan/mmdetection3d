_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_aicv-8class-voxel0.25-80m.py',
    '../_base_/datasets/hesai90-3d-8class-80m.py',
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
            ranges=[[-80, -80, -1.78, 80, 80, -1.78],
                    [-80, -80, -1.78, 80, 80, -1.78],
                    [-80, -80, -1.87, 80, 80, -1.87],
                    [-80, -80, -1.90, 80, 80, -1.90],
                    [-80, -80, -1.80, 80, 80, -1.80],
                    [-80, -80, -1.95, 80, 80, -1.95],
                    [-80, -80, -1.94, 80, 80, -1.94],
                    [-80, -80, -1.93, 80, 80, -1.93]],
            sizes=[[4.27, 1.87, 1.56],      # Car
                   [6.71, 2.31, 2.43],      # Bus
                   [1.80, 0.76, 1.10],      # NonMot
                   [1.85, 0.87, 1.56],      # Cyclist
                   [0.55, 0.60, 1.65],      # Pedestrian
                   [0.37, 0.37, 0.66],      # TrafficCone
                   [2.24, 0.56, 0.79],      # Barrier
                   [2.73, 2.29, 0.75]],     # Others
            rotations=[0.7854, 2.3562],
            reshape_out=False)))

# runtime settings
lr_config = dict(step=[56, 59])
runner = dict(max_epochs=60)
evaluation = dict(interval=4)
