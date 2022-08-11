# model settings
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
voxel_size = [0.2, 0.2, 6]
model = dict(
    type='MVXFasterRCNN',
    pts_voxel_layer=dict(
        max_num_points=64,
        point_cloud_range=[-60, -60, -3, 60, 60, 3],
        voxel_size=voxel_size,
        max_voxels=(30000, 40000)),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[-60, -60, -3, 60, 60, 3],
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[600, 600]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=[64, 128, 256],
        out_channels=256,
        start_level=0,
        num_outs=3),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=8,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-60, -60, -1.7, 60, 50, -1.7]],
            scales=[1, 2, 4],
            sizes=[
                [2.5981, 0.8660, 1.],  # 1.5 / sqrt(3)
                [1.7321, 0.5774, 1.],  # 1 / sqrt(3)
                [1., 1., 1.],
                [0.4, 0.4, 1],
            ],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,  # -pi / 4
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.55,       # 正样本的iou阈值
                neg_iou_thr=0.3,        # 负样本的iou阈值
                min_pos_iou=0.3,        # 正样本的iou最小值。如果assign给ground truth的anchors中最大的IOU低于0.3，则忽略所有的anchors，否则保留最大IOU的anchor
                ignore_iof_thr=-1),     # 忽略bbox的阈值，当ground truth中包含需要忽略的bbox时使用，-1表示不忽略
            allowed_border=0,           # 是否允许在bbox周围外扩一定的像素
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            pos_weight=-1,              # 正样本权重，-1表示不改变原始的权重
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,    # 在所有的fpn层内做nms
            nms_pre=1000,               # 在nms之前保留的的得分最高的proposal数量
            nms_thr=0.2,                # nms阈值
            score_thr=0.05,
            min_bbox_size=0,            # 最小bbox尺寸
            max_num=500)))              # 在nms之后保留的的得分最高的proposal数量
