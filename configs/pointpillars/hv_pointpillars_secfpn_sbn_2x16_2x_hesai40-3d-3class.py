_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_aicv.py',
    '../_base_/datasets/hesai40-3d-3class.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

voxel_size = [0.32, 0.32, 6]
model = dict(
    type='MVXFasterRCNN',
    pts_voxel_layer=dict(
        max_num_points=20,
        point_cloud_range=[-74.88, -74.88, -4, 74.88, 74.88, 2],
        voxel_size=voxel_size,
        max_voxels=(32000, 32000)),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[-74.88, -74.88, -4, 74.88, 74.88, 2],
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)))

runner = dict(max_epochs=60)
evaluation = dict(interval=10)