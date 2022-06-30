_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_aicv-voxel0.2-60m.py',
    '../_base_/datasets/hesai40-bp-3d-60m.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

runner = dict(max_epochs=60)
evaluation = dict(interval=4)