optimizer=dict(
    type='AdamW',
    param_dict_type='default',
    lr=1e-4,
    lr_backbone=1e-4,
    weight_decay=0.05,
)

scheduler=dict(
    type='PolyLR',
    total_iters=300*512,  # 30 epoch
    power=0.9,
)
