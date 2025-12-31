# From baseline3dseg_maxnorm10_rdrop2d
_base_ = [
    "../datasets/instance_segmentation_3d/scannet.py",  # basic dataset
    "../models/base_3d.py",  # basic modeling
    "../schedules/adamw_polylr_3d.py",  # basic schedule
]
voxel_size = 0.02
num_instance_classes = 18
num_semantic_classes = 20

# 1. Model configurations.
# 1.1 for spconv backbone and scannet 20 classes.
pointcloud_backbone_cfg = dict(
    type='SpConvUNet',
    num_planes=[32 * (i + 1) for i in range(5)],
    return_blocks=True,
    voxel_size = voxel_size,
    mode_fuse_2d_feat = "early_fusion")
_base_.num_instance_classes = num_instance_classes
_base_.num_semantic_classes = num_semantic_classes
_base_.model.pointcloud_backbone_cfg = pointcloud_backbone_cfg
_base_.model.num_classes = num_instance_classes
_base_.model.decoder_cfg.num_instance_classes = num_instance_classes
_base_.model.decoder_cfg.num_semantic_classes = num_semantic_classes
_base_.model.decoder_cfg.in_channels = 32
_base_.model.criterion_cfg.inst_criterion.num_classes = num_instance_classes
_base_.model.criterion_cfg.num_semantic_classes = num_semantic_classes
_base_.model.criterion_cfg.sem_criterion.ignore_index = num_semantic_classes
# 1.2 for extra designs.
_base_.model.decoder_cfg.add_box_size_pred = True
_base_.model.add_positional_embedding = True
_base_.model.mode_3d_center = 'median'
_base_.model.decoder_cfg.add_positional_embedding = True
_base_.model.decoder_cfg.pos_type = 'sine'
_base_.model.decoder_cfg.temperature = 20
_base_.model.pointcloud_backbone_cfg.add_positional_embedding = True
_base_.model.criterion_cfg.inst_criterion.matcher = dict(
                                type='SparseMatcher',
                                    costs=[
                                        dict(type='QueryClassificationCost', weight=0.5),
                                        dict(type='MaskBCECost', weight=1.0),
                                        dict(type='MaskDiceCost', weight=1.0),
                                        dict(type='CenterL1Cost', weight=0.5),
                                        dict(type='SizeL1Cost', weight=0.5)],
                                topk=1)
_base_.model.criterion_cfg.inst_criterion.loss_weight = [0.5, 1.0, 1.0, 0.5, 0.5, 0.5]
_base_.model.decoder_cfg.box_modulate_ca = True
_base_.model.filter_outofbox_points_eval = True #! use box to filter mask

# 2. More data configurations.
_base_.scannet_train_mask.transform_cfg.transform_cfg.voxel_size = voxel_size
_base_.scannet_val_mask.transform_cfg.transform_cfg.voxel_size = voxel_size
_base_.scannet_train_mask.dropout_rate_2dfeats = 0.1
data = dict(
    train_main=[
        _base_.scannet_train_mask,
    ],
    eval_main=[
        _base_.scannet_val_mask,
    ],
    train_extras=None,
    train_batch_size=4,
    pin_memory=False,
    num_workers=8,
    sync_scale=True)

# 3. Evaluation configurations.
evaluations = ["scannet_instance_seg"]
metric_meta = dict(
    label2cat=_base_.label2cat,
    ignore_index=[_base_.num_semantic_classes],
    classes=_base_.class_names + ['unlabeled'],
    dataset_name='ScanNet')
evaluator_cfg = dict(
    stuff_class_inds=[0, 1], 
    thing_class_inds=list(range(2, _base_.num_semantic_classes)),
    min_num_points=1, 
    id_offset=2**16,
    sem_mapping=_base_.sem_mapping,
    inst_mapping=_base_.inst_mapping,
    metric_meta=metric_meta,
    dataset="scannet")

# 4. Optimization configurations.
amp = False
use_ema = False
ema_decay = 0.9997
ema_epoch = 0  # which epoch to start using ema
clip_max_norm = 10
num_iterations = 150000  # 300 * 500
eval_step = 300 * 500  # 300 * 500
save_step = 300 * 16
print_freq = 10
