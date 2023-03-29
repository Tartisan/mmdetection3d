class_names = ['Car', 'Pedestrian', 'Bicycle', 'TrafficCone']
point_cloud_range = [-84.8, -84.8, -3.5, 84.8, 84.8, 3.5]

# model config
voxel_size = [0.2, 0.2, 7.0]
model = dict(
    type='CPDetTwoStage',
    voxel_layer=dict(
        max_num_points=-1, 
        voxel_size=voxel_size, 
        max_voxels=(-1, -1),
        point_cloud_range=point_cloud_range),
    voxel_encoder=dict(
        type='DynamicPillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        input_norm=True),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(848, 848)),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[1, 2, 2],
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
            dict(num_class=4, class_names=['Car', 'Pedestrian', 'Bicycle', 'TrafficCone'])
        ],
        common_heads=dict(
            # (output_channel, num_conv)
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-90, -90, -6.0, 90, 90, 6.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=2,
            voxel_size=voxel_size[:2],
            code_size=7,
            pc_range=point_cloud_range[:2]),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    roi_head=dict(
        type='CenterPointRoIHead',
        bev_feature_extractor_cfg=dict(
            type='BEVFeatureExtractor',
            pc_start=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
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
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            grid_size=[848, 848, 1],
            voxel_size=voxel_size,
            out_size_factor=2,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            point_cloud_range=point_cloud_range),
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
            post_center_limit_range=[-90, -90, -10.0, 90, 90, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            pc_range=point_cloud_range[:2],
            out_size_factor=2,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=4096,
            post_max_size=500,
            nms_thr=0.01),
        rcnn=dict()))


# optimizer
lr = 0.0000009
optimizer = dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# 默认初始 lr=0.0018，训练到 40% 时涨到 10 倍，然后下降到 1e-4 倍
lr_config = dict(policy='cyclic', target_ratio=(10, 1e-4), 
                 cyclic_times=1, step_ratio_up=0.4)
momentum_config = dict(policy='cyclic', target_ratio=(0.85 / 0.95, 1), 
                       cyclic_times=1, step_ratio_up=0.4)

runner = dict(type='EpochBasedRunner', max_epochs=6)


# dataset settings
dataset_type = 'AicvDataset'
data_root = 'data/hesai40/mb-shougang/kitti_format/'
file_client_args = dict(backend='disk')
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'mb-shougang_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_min_points=dict(
            Car=5, Pedestrian=5, Bicycle=5, TrafficCone=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Bicycle=10, TrafficCone=5),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3],
        file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train = dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'mb-shougang_infos_train.pkl',
            split='training',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR')),
    val = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'mb-shougang_infos_val.pkl',
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'mb-shougang_infos_val.pkl',
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

evaluation = dict(interval=6, pipeline=eval_pipeline)

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None # 'work_dirs/cpdet_0.2_84.8m_dv_second_secfpn_dw2_4cls_1tsk/cpdet_0.2_84.8m_dv_second_secfpn_dw2_4cls_1tsk_db_12e.pth'
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
