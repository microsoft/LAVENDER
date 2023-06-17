_base_ = [
    './swin_violet.py', './default_runtime.py'
]
model=dict(backbone=dict(patch_size=(2,4,4), drop_path_rate=0.2))

