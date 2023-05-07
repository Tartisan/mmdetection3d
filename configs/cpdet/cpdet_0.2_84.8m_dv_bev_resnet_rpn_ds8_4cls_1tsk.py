class_names = ['Car', 'Pedestrian', 'Bicycle', 'TrafficCone']
point_cloud_range = [-84.8, -84.8, -3.5, 84.8, 84.8, 3.5]

# model config
voxel_size = [0.2, 0.2, 7.0]
model = dict(
    type='CPDet',
    pts_voxel_layer=dict(
        type='CustomVoxelization',
        max_num_points=-1,
        voxel_size=voxel_size,
        max_voxels=(-1, -1),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='DynamicPillarFeatureNet',
        in_channels=4,
        feat_channels=[48],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        input_norm=True),
    pts_middle_encoder=dict(
        type="PointPillarsScatterExpand",
        in_channels=48,
        expand_bev_channels=16,
        out_channels=64,
        output_shape=(848, 848),
        remove_intensity=False),
    pts_backbone=dict(
        type='CustomResNet',
        depth=18,
        num_stages=4,
        base_channels=64,
        out_indices=(2, 3),
        strides=(2, 2, 2, 2),
        with_cp=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        style='pytorch'),
    pts_neck=dict(
        type="RPNV2",
        layer_nums=[3, 3],
        ds_layer_strides=[1, 2],
        ds_num_filters=[512, 512],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=[256, 512]),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([128, 128, 128, 128]),
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
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=7,
            pc_range=point_cloud_range[:2]),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[848, 848, 1],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-90, -90, -10.0, 90, 90, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            pc_range=point_cloud_range[:2],
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=4096,
            post_max_size=500,
            nms_thr=0.01)))


# optimizer
lr = 1.8e-04
optimizer = dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# 默认初始 lr=0.0018，训练到 40% 时涨到 10 倍，然后下降到 1e-4 倍
lr_config = dict(policy='cyclic', target_ratio=(10, 1e-4), 
                 cyclic_times=1, step_ratio_up=0.4)
momentum_config = dict(policy='cyclic', target_ratio=(0.85 / 0.95, 1), 
                       cyclic_times=1, step_ratio_up=0.4)

runner = dict(type='EpochBasedRunner', max_epochs=24)


# dataset settings
dataset_type = 'AicvDataset'
data_root = 'data/hesai40/mb-bp/kitti_format/'
file_client_args = dict(backend='disk')
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'mb-bp_dbinfos_train.pkl',
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
    dict(type='ObjectSample', db_sampler=db_sampler),
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
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'mb-bp_infos_train.pkl',
            split='training',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'mb-bp_infos_val.pkl',
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'mb-bp_infos_val.pkl',
        split='training',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

evaluation = dict(interval=24, pipeline=eval_pipeline)

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
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
