_base_ = [
    '../_base_/datasets/hesai90-3d-4class-80m.py',
    '../_base_/models/centerpoint_02pillar_second_secfpn_aicv-4class-voxel0.2-80m.py',
    '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-80.0, -80.0, -3.0, 80.0, 80.0, 3.0]

model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))


data = dict(samples_per_gpu=8, workers_per_gpu=4)
evaluation = dict(interval=1)


lr = 0.0018
optimizer = dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# 初始 lr=0.0018，训练到 40%*max_epochs 时涨到最大 0.018，然后缓慢下降到 1.8e-7
lr_config = dict(policy='cyclic', target_ratio=(10, 1e-4), 
                 cyclic_times=1, step_ratio_up=0.4,)
momentum_config = dict(policy='cyclic', target_ratio=(0.85 / 0.95, 1), 
                       cyclic_times=1, step_ratio_up=0.4)

runner = dict(type='EpochBasedRunner', max_epochs=80)

