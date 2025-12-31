import segdino3d.datasets.transform.point_cloud_transforms as T
from typing import List
from segdino3d import TRANSFORMS


@TRANSFORMS.register_module()
def Scannet200Transforms(scene_set: str, voxel_size=0.02, debug=False, 
                   ) -> T.Compose3D:
    """Build COCOTransforms.

    rgs:
        scene_set (str): The image set to use. Must be one of 'train', 'val', 'eval_debug',
            'train_reg', or 'test'.

    Returns:
        T.Compose: A set of PyTorch transforms for the scannet200 dataset.
    """

    color_mean = (
        0.47793125906962 * 255,
        0.4303257521323044 * 255,
        0.3749598901421883 * 255)
    color_std = (
        0.2834475483823543 * 255,
        0.27566157565723015 * 255,
        0.27018971370874995 * 255)
    if scene_set == 'train':
        return T.Compose3D([
            T.CustomRandomFlip3D(
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5,),
            T.CustomGlobalRotScaleTrans(
                rot_range=[-3.14, 3.14],
                scale_ratio_range=[0.8, 1.2],
                translation_std=[0.1, 0.1, 0.1]),
            T.NormalizePointsColor(
                color_mean=color_mean,
                color_std=color_std),
            T.ElasticTransfrom(
                gran=[6, 20],
                mag=[40, 160],
                voxel_size=voxel_size,
                p=0.5),
            T.ToTensor(),
        ])

    elif scene_set in ['val', 'test']:
        return T.Compose3D([
            T.NormalizePointsColor(
                color_mean=color_mean,
                color_std=color_std),
            T.ToTensor(),
        ])

    else:
        raise ValueError(f'unknown {scene_set}')

