_base_ = [
    "./Asheep_weight_segdetbigdataset.py",
    "./sheep_weight_runtime.py"
]

total_epochs = 100
batch_size = 64

# model settings
model = dict(
    type="TopDown",
    backbone=dict(
        type="LiteHRNet",
        extra=dict(
            stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(2, 4, 2),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=("LITE", "LITE", "LITE"),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (40, 80),
                    (40, 80, 160),
                    (40, 80, 160, 320),
                ),
            ),
            with_head=False,
        ),
        in_channels=4,
    ),
    keypoint_head=dict(
        type="WeightHead5",
        in_channels=40,
        input_size=(64, 64),
        loss_keypoint=dict(type="MSELoss"),
    ),
    train_cfg=dict(),
    test_cfg=dict(flip_test=False),
)
