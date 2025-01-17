voxel_size_ = [0.2, 0.2, 8]
point_cloud_range_ = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
model = dict(
    type='CenterPointTwoStage',
    voxel_layer=dict(
        max_num_points=20,
        voxel_size=voxel_size_,
        max_voxels=(30000, 40000),
        point_cloud_range=point_cloud_range_),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size_,
        point_cloud_range=point_cloud_range_,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(512, 512)),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    rpn_head=dict(
        type='CenterHead',
        in_channels=sum([128, 128, 128]),
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range_[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size_[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    roi_head=dict(
        type='CenterPointRoIHead',
        bev_feature_extractor_cfg=dict(
            type='BEVFeatureExtractor',
            pc_start=point_cloud_range_[:2],
            voxel_size=voxel_size_[:2],
            downsample_stride=1,
        ),
        bbox_head=dict(
            type='CenterPointBBoxHead',
            input_channels=128 * 3 * 5,
            shared_fc=[256, 256],
            cls_fc=[256, 256],
            reg_fc=[256, 256],
            dp_ratio=0.3,
            code_size=7,
            num_classes=1,
            loss_reg=dict(type='L1Loss', reduction='mean', loss_weight=1.0),
            loss_cls=dict(
                type='CrossEntropyLoss',
                reduction='mean',
                use_sigmoid=True,
                loss_weight=1.0)),
    ),
    train_cfg=dict(
        rpn=dict(
            max_objs=500,
            dense_reg=1,
            grid_size=[512, 512, 1],
            point_cloud_range=point_cloud_range_,
            voxel_size=voxel_size_,
            out_size_factor=4,
            gaussian_overlap=0.1,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                pos_iou_thr=0.01,
                neg_iou_thr=0.01,
                min_pos_iou=0.01,
                ignore_iof_thr=-1,
                match_low_quality=False),
            sampler=dict(
                type='IoUNegPiecewiseSampler',
                num=128,
                pos_fraction=0.5,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True),
            cls_pos_thr=0.7,
            cls_neg_thr=0.25,
            reg_pos_thr=0.01)),
    test_cfg=dict(
        rpn=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            pc_range=point_cloud_range_[:2],
            out_size_factor=4,
            voxel_size=voxel_size_[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2),
        rcnn=dict()))
