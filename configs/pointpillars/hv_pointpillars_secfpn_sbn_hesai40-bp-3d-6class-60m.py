_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_aicv-6class-voxel0.2-60m.py',
    '../_base_/datasets/hesai40-bp-3d-6class-60m.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

# runtime settings
lr_config = dict(step=[20, 23])
runner = dict(max_epochs=24)
evaluation = dict(interval=4)