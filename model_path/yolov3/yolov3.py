_base_ = '.\yolov3_d53_8xb8-ms-608-273e_coco.py'

data_root = r'E:\3_Entrepreneurship\XZT\Dataset\three_diseases'
dataset_type = 'CocoDataset'
backend_args = None

train_dataloader = dict(
dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + r'\Annotations\train_food.json',
        data_prefix=dict(img=r'E:\3_Entrepreneurship\XZT\Dataset\three_diseases\JPEGImages'))
)

val_dataloader = dict(
dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + r'\Annotations\val_food.json',
        data_prefix=dict(img=r'E:\3_Entrepreneurship\XZT\Dataset\three_diseases\JPEGImages')))



val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + r'\Annotations\val_food.json',
    metric='bbox',
    backend_args=backend_args)


test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# inference on test dataset and
# format the output results for submission.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + r'\Annotations\test_food.json',
        data_prefix=dict(img=r'E:\3_Entrepreneurship\XZT\Dataset\three_diseases\JPEGImages'),
        test_mode=True,
        # pipeline=test_pipeline))
))

test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=True,
    ann_file=data_root + r'\Annotations\test_food.json',
    outfile_prefix=r'E:\3_Entrepreneurship\XZT\ShangYiDemo\ShangYiDemo\src\result_imgs')



