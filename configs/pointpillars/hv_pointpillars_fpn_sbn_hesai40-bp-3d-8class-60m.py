_base_ = [
    '../_base_/models/hv_pointpillars_fpn_aicv-8class-voxel0.2-60m.py',
    '../_base_/datasets/hesai40-bp-3d-8class-60m.py',
    '../_base_/default_runtime.py',
]

data = dict(samples_per_gpu=4, workers_per_gpu=4)

# optimizer
# This schedule is mainly used by models on nuScenes dataset
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',              # 优化策略
    warmup='linear',            # 初始的学习率增加的策略，linear为线性增加
    warmup_iters=1000,          # 在初始的1000次迭代中学习率逐渐增加
    warmup_ratio=1.0 / 1000,    # 起始的学习率
    step=[68, 71])              # 在第68和71个epoch时降低学习率
momentum_config = None              
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=60)


evaluation = dict(interval=4)