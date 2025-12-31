

num_instance_classes = 198  # for instance segmentation, we use 198 classes, excluding the stuffs.
num_semantic_classes = 200  # for panoptic / semantic segmentation, we use 200 classes, including the stuffs.

pointcloud_backbone_cfg = dict(
    type='Res16UNet34C',
    in_channels=256 + 3,
    out_channels=96,
    config=dict(
        dilations=[1, 1, 1, 1],
        conv1_kernel_size=5,
        bn_momentum=0.02)
)
decoder_cfg = dict(
    type='ScanNetQueryDecoder',
    add_dinox_query_ca = True,
    add_dinox_query_ca_mask = True,
    dinox_query_ca_mask_threshold = 0.2,
    num_layers=6,
    num_instance_queries=0,
    num_semantic_queries=0,
    num_instance_classes=num_instance_classes,
    num_semantic_classes=num_semantic_classes,
    num_semantic_linears=1,
    in_channels=96,
    d_model=256,
    num_heads=8,
    hidden_dim=1024,
    dropout=0.0,
    activation_fn='gelu',
    iter_pred=True,
    attn_mask=True,
    fix_attention=True,
    objectness_flag=False
)
criterion_cfg = dict(
    type='ScanNetUnifiedCriterion',
    num_semantic_classes=num_semantic_classes,
    sem_criterion=dict(
        type='ScanNetSemanticCriterion',
        ignore_index=num_semantic_classes,
        loss_weight=0.5),
    inst_criterion=dict(
        type='InstanceCriterion',
        matcher=dict(
            type='SparseMatcher',
            costs=[
                dict(type='QueryClassificationCost', weight=0.5),
                dict(type='MaskBCECost', weight=1.0),
                dict(type='MaskDiceCost', weight=1.0)],
            topk=1),
        loss_weight=[0.5, 1.0, 1.0, 0.5],
        num_classes=num_instance_classes,
        non_object_weight=0.1,
        fix_dice_loss_weight=True,
        iter_matcher=True,
        fix_mean_loss=True))
neck_cfg = None
transformer_cfg = None
text_encoder_cfg = None
model = dict(
    type='Baseline3D',
    num_classes=num_instance_classes,
    pointcloud_backbone_cfg=pointcloud_backbone_cfg,
    decoder_cfg=decoder_cfg,
    text_encoder_cfg=text_encoder_cfg,
    criterion_cfg=criterion_cfg,
    query_thr=0.5,
    test_cfg=dict(
        topk_insts=600,
        inst_score_thr=0.0,
        pan_score_thr=0.5,
        npoint_thr=100,
        obj_normalization=True,
        sp_score_thr=0.4,
        nms=True,
        matrix_nms_kernel='linear',
        stuff_classes=[0, 1])
)