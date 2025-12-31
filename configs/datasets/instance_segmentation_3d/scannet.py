
_base_ = '../../transforms/scannet200_transform.py'


scannet_train_mask = dict(
    type='ScanNetInstanceSeg3D',
    scene_set="train",
    root_scenes="data/scannet", #! change to your data path
    use_super_points=True,
    root_points_2dfeats="data/features_2d/scannet",  #! change to your 2D feature path
    transform_cfg=dict(
        type='Segment3DTransform',
        preparer_cfg=dict(
            type='InstanceSeg3DDataPreparer'
        ),
        transform_cfg=_base_.scannet200_transform_train,
    ),
    mode_fuse_multi_scale_2d_feats="mean",
    dataset_type="scannet_train_mask3d",
)

scannet_val_mask = dict(
    scene_set="val",
    root_scenes="data/scannet", #! change to your data path
    use_super_points=True,
    root_points_2dfeats="data/features_2d/scannet",  #! change to your 2D feature path
    transform_cfg=dict(
        type='Segment3DTransform',
        preparer_cfg=dict(
            type='InstanceSeg3DDataPreparer'
        ),
        transform_cfg=_base_.scannet200_transform_val,
    ),
    mode_fuse_multi_scale_2d_feats="mean",
    dataset_type="scannet_val_mask3d",
)

sem_mapping = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
]
class_names = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
    'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
    'otherfurniture'
]
inst_mapping = sem_mapping[2:]
label2cat = {i: name for i, name in enumerate(class_names + ['unlabeled'])}
