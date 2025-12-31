# From baseline3dseg_maxnorm10_rdrop2d
_base_ = [
    "../datasets/instance_segmentation_3d/scannet200.py",  # basic dataset
    "../models/base_3d.py",  # basic modeling
    "../schedules/adamw_polylr_3d.py",  # basic schedule
]
voxel_size = 0.02

# 1. Model configurations.
_base_.model.pointcloud_backbone_cfg.voxel_size = voxel_size
_base_.model.pointcloud_backbone_cfg.mode_fuse_2d_feat = "only_rgb"
_base_.model.decoder_cfg.add_dinox_query_ca = False

# 2. More data configurations.
_base_.scannet200_train_mask.transform_cfg.transform_cfg.voxel_size = voxel_size
_base_.scannet200_val_mask.transform_cfg.transform_cfg.voxel_size = voxel_size
_base_.scannet200_train_mask.dropout_rate_2dfeats = 0.7
data = dict(
    train_main=[
        _base_.scannet200_train_mask,
    ],
    eval_main=[
        _base_.scannet200_val_mask,
    ],
    train_extras=None,
    train_batch_size=4,
    pin_memory=False,
    num_workers=8,
    sync_scale=True)

# 3. Evaluation configurations.
evaluations = ["scannet200_instance_seg"]
metric_meta = dict(
    label2cat=_base_.label2cat,
    ignore_index=[_base_.num_semantic_classes],
    classes=_base_.class_names + ['unlabeled'],
    dataset_name='ScanNet200')
evaluator_cfg = dict(
    stuff_class_inds=[0, 1], 
    thing_class_inds=list(range(2, _base_.num_semantic_classes)),
    min_num_points=1, 
    id_offset=2**16,
    sem_mapping=_base_.sem_mapping,
    inst_mapping=_base_.inst_mapping,
    metric_meta=metric_meta)

# 5. Optimization configurations.
amp = False
use_ema = False
ema_decay = 0.9997
ema_epoch = 0  # which epoch to start using ema
clip_max_norm = 10
num_iterations = 300 * 129
eval_step = 300 * 129
save_step = 300 * 4
print_freq = 10

_base_.scheduler.total_iters = num_iterations  # 129 epoch