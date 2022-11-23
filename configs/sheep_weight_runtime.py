#!/usr/bin/env python3

log_level = "INFO"
load_from = None
resume_from = None
dist_params = dict(backend="nccl")
workflow = [("train", 1)]
checkpoint_config = dict(interval=30)


evaluation = dict(
    interval=1,
    save_best="AbsErrorP",
    less_keys=("ErrorP", "AbsErrorP", "SingleMaxP"),
)
custom_imports = dict(
    imports=["mmpose.models.heads.reset_checkpoint_bestscore"],
    allow_failed_imports=False,
)
custom_hooks_config = [
    dict(type="ResetCheckpointBestScore", init_value="inf", priority="NORMAL")
]

optimizer = dict(
    type="Adam",
    lr=1e-2,
)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy="step",
#     warmup="linear",
#     warmup_iters=100,
#     warmup_ratio=0.001,
#     step=500,
# )
# lr_config = dict(policy='CosineAnnealing', min_lr_ratio=1e-5)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4)

log_config = dict(
    interval=1,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")],
)
