
scannet200_transform_train = dict(
    type='Scannet200Transforms',  # some data augmentations.
    scene_set='train',
    voxel_size=0.02,
    debug=False,
)

scannet200_transform_val = dict(
    type='Scannet200Transforms',
    scene_set='val',
    voxel_size=0.02,
)
