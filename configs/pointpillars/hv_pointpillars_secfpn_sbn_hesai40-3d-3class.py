_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_aicv-3class.py',
    '../_base_/datasets/hesai40-3d-3class.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

runner = dict(max_epochs=120)
evaluation = dict(interval=4)