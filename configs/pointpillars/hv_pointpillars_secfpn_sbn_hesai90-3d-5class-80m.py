_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_aicv-5class-voxel0.25-80m.py',
    '../_base_/datasets/hesai90-3d-5class-80m.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

# runtime settings
lr_config = dict(step=[116, 119])
runner = dict(max_epochs=120)
evaluation = dict(interval=4)
