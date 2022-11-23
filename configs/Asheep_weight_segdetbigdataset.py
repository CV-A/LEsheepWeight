#!/usr/bin/env python3

batch_size = 16
total_epochs = 50

dataset_info = dict(
    dataset_name='SheepWeightDataset',
    paper_info=" ",
)

train_pipeline = [
    dict(type="LoadImageDepthFromFile"),
    dict(type="ToTensor"),
    dict(type="SheepRandomHorizontalFlip"),
    dict(type="SheepRandomVerticalFlip"),
    dict(type="SheepRandomRotation", degrees=30),
    dict(type="SheepResize", size=(256, 256)),
    dict(
        type="NormalizeTensor",
        mean=[0.485, 0.456, 0.406, 0],
        std=[0.229, 0.224, 0.225, 1],
    ),
    dict(
        type="SheepCollect",
        keys=["img", "target"],
        meta_keys=[
            "image_file",
            "target",
        ],
    ),
]

val_pipeline = [
    dict(type="LoadImageDepthFromFile"),
    # dict(type="TopDownAffine"),
    dict(type="Resize", size=(256, 256)),
    dict(type="ToTensor"),
    dict(
        type="NormalizeTensor",
        mean=[0.485, 0.456, 0.406, 0],
        std=[0.229, 0.224, 0.225, 1],
    ),
    dict(
        type="SheepCollect",
        keys=["img", "target"],
        meta_keys=[
            "image_file",
            "target",
        ],
    ),
]

test_pipeline = val_pipeline
data_root = "data/sheepweight/sideweight"
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=16,
    shuffle=True,
    val_dataloader=dict(samples_per_gpu=batch_size),
    test_dataloader=dict(samples_per_gpu=batch_size),
    train=dict(
        type="SheepWeightDataset",
        dataset_info = dict(
            dataset_name='SheepWeightDataset',
            paper_info="",
        ),
        csv_file=f"{data_root}/weight_annotation.csv",
        img_prefix=f"{data_root}/trainsegdet",
        pipeline=train_pipeline,
    ),
    val=dict(
        type="SheepWeightDataset",
        dataset_info = dict(
            dataset_name='SheepWeightDataset',
            paper_info="",
        ),
        csv_file=f"{data_root}/weight_annotation.csv",
        img_prefix=f"{data_root}/valsegdet",
        pipeline=val_pipeline,
    ),
    test=dict(
        type="SheepWeightDataset",
        dataset_info = dict(
            dataset_name='SheepWeightDataset',
            paper_info="",
        ),
        csv_file=f"{data_root}/weight_annotation.csv",
        img_prefix=f"{data_root}/valsegdet",
        pipeline=val_pipeline,
    ),
)
