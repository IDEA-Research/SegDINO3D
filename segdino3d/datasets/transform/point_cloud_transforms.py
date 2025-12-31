import os
import random
import numpy as np
from typing import Tuple, List
from typing import Iterator, Optional, Sequence, Union
import scipy

import PIL
import torch
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F

from mmdet.datasets.transforms import RandomFlip
from mmdet3d.datasets.transforms import GlobalRotScaleTrans
from mmdet3d.structures.bbox_3d.utils import rotation_3d_in_axis


class ToTensor(object):

    def __call__(self, points, target):
        return torch.tensor(points), target


class Compose3D(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points, target):
        for t in self.transforms:
            points, target = t(points, target)
        return points, target
    

class CustomRandomFlip3D(RandomFlip):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Required Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Modified Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Added Keys:

    - points (np.float32)
    - pcd_trans (np.float32)
    - pcd_rotation (np.float32)
    - pcd_rotation_angle (np.float32)
    - pcd_scale_factor (np.float32)

    Args:
        sync_2d (bool): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float): The flipping probability
            in vertical direction. Defaults to 0.0.
        flip_box3d (bool): Whether to flip bounding box. In most of the case,
            the box should be fliped. In cam-based bev detection, this is set
            to False, since the flip of 2D images does not influence the 3D
            box. Defaults to True.
    """

    def __init__(self,
                 flip_ratio_bev_horizontal: float = 0.0,
                 flip_ratio_bev_vertical: float = 0.0,
                 **kwargs) -> None:
        # `flip_ratio_bev_horizontal` is equal to
        # for flip prob of 2d image when
        # `sync_2d` is True
        super(CustomRandomFlip3D, self).__init__(
            prob=flip_ratio_bev_horizontal, direction='horizontal', **kwargs)
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self,
                            points,
                            targets: dict,
                            direction: str = 'horizontal') -> None:
        """Flip 3D data randomly.

        `random_flip_data_3d` should take these situations into consideration:

        - 1. LIDAR-based 3d detection
        - 2. LIDAR-based 3d segmentation
        - 3. vision-only detection
        - 4. multi-modality 3d detection.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Defaults to 'horizontal'.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are
            updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        if direction == 'horizontal':
            points[:, 0] = -points[:, 0]
        elif direction == 'vertical':
            points[:, 1] = -points[:, 1]
        else:
            raise NotImplementedError

        if not targets["extra_features"]["query2d_pos"] is None:
            if direction == 'horizontal':
                targets["extra_features"]["query2d_pos"][:, 0] = -targets["extra_features"]["query2d_pos"][:, 0]
            elif direction == 'vertical':
                targets["extra_features"]["query2d_pos"][:, 1] = -targets["extra_features"]["query2d_pos"][:, 1]
        return points, targets

    def __call__(self, points, targets: dict) -> dict:
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
            'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
            into result dict.
        """
        if 'pcd_horizontal_flip' not in targets:
            flip_horizontal = True if np.random.rand(
            ) < self.flip_ratio_bev_horizontal else False
            targets['pcd_horizontal_flip'] = flip_horizontal
        if 'pcd_vertical_flip' not in targets:
            flip_vertical = True if np.random.rand(
            ) < self.flip_ratio_bev_vertical else False
            targets['pcd_vertical_flip'] = flip_vertical

        if targets['pcd_horizontal_flip']:
            points, targets = self.random_flip_data_3d(points, targets, 'horizontal')
        if targets['pcd_vertical_flip']:
            points, targets = self.random_flip_data_3d(points, targets, 'vertical')
        return points, targets

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        repr_str += f' flip_ratio_bev_horizontal={self.flip_ratio_bev_horizontal})'
        return repr_str
    

class CustomGlobalRotScaleTrans(GlobalRotScaleTrans):
    """Apply global rotation, scaling and translation to a 3D scene.

    Required Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Modified Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Added Keys:

    - points (np.float32)
    - pcd_trans (np.float32)
    - pcd_rotation (np.float32)
    - pcd_rotation_angle (np.float32)
    - pcd_scale_factor (np.float32)

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of
            translation noise applied to a scene, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0].
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range: List[float] = [-0.78539816, 0.78539816],
                 scale_ratio_range: List[float] = [0.95, 1.05],
                 translation_std: List[int] = [0, 0, 0],
                 shift_height: bool = False) -> None:
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'

        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        self.shift_height = shift_height

    def _trans_points(self, points, targets: dict) -> None:
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans'
            and `gt_bboxes_3d` is updated in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T.astype(np.float32)

        points[:, :3] = points[:, :3] + trans_factor
        if "query2d_pos" in targets["extra_features"]:
            targets["extra_features"]["query2d_pos"] = targets["extra_features"]["query2d_pos"] + \
                targets["extra_features"]["query2d_pos"].new_tensor(trans_factor)
        return points, targets

    def _rot_points(self, points, targets: dict) -> None:
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation'
            and `gt_bboxes_3d` is updated in the result dict.
        """
        def rotate(
                points: Tensor,
                rotation: Union[Tensor, np.ndarray, float],
                axis: Optional[int] = None) -> Tensor:
            """Rotate points with the given rotation matrix or angle.

            Args:
                rotation (Tensor or np.ndarray or float): Rotation matrix or angle.
                axis (int, optional): Axis to rotate at. Defaults to None.

            Returns:
                Tensor: Rotation matrix.
            """
            if not isinstance(rotation, Tensor):
                rotation = points.new_tensor(rotation)
            assert rotation.shape == torch.Size([3, 3]) or rotation.numel() == 1, \
                f'invalid rotation shape {rotation.shape}'

            if axis is None:
                axis = 0

            if rotation.numel() == 1:
                rotated_points, rot_mat_T = rotation_3d_in_axis(
                    points[:, :3][None], rotation, axis=axis, return_mat=True)
                points[:, :3] = rotated_points.squeeze(0)
                rot_mat_T = rot_mat_T.squeeze(0)
            else:
                # rotation.numel() == 9
                points[:, :3] = points[:, :3] @ rotation
                rot_mat_T = rotation

            return points, rot_mat_T
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        # if no bbox in input_dict, only rotate points
        points, rot_mat_T = rotate(points, noise_rotation, axis=2)
        if "query2d_pos" in targets["extra_features"]:
            targets["extra_features"]["query2d_pos"], noise_rotation = rotate(targets["extra_features"]["query2d_pos"], noise_rotation, axis=2)

        targets['pcd_rotation'] = rot_mat_T
        targets['pcd_rotation_angle'] = noise_rotation
        return points, targets

    def _scale_points(self, points, targets: dict) -> None:
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points' and
            `gt_bboxes_3d` is updated in the result dict.
        """
        scale = targets['pcd_scale_factor']
        points[:, :3] *= scale
        if self.shift_height:
            raise NotImplementedError
        if "query2d_pos" in targets["extra_features"]:
            query2d_pos = targets["extra_features"]["query2d_pos"]
            query2d_pos = query2d_pos * scale
            targets["extra_features"]["query2d_pos"] = query2d_pos
        return points, targets

    def _random_scale(self, input_dict: dict) -> None:
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor'
            are updated in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor
        return input_dict

    def __call__(self, points, targets: dict) -> dict:
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans' and `gt_bboxes_3d` are updated
            in the result dict.
        """
        points, targets = self._rot_points(points, targets)
        targets = self._random_scale(targets)
        points, targets = self._scale_points(points, targets)
        points, targets = self._trans_points(points, targets)
        return points, targets


class NormalizePointsColor(object):
    """Just add color_std parameter.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
        color_std (list[float]): Std color of the point cloud.
            Default value is from SPFormer preprocessing.
    """

    def __init__(self, color_mean, color_std=127.5):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, points, targets: dict):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
            Updated key and value are described below.
                - points (:obj:`BasePoints`): Points after color normalization.
        """
        assert points.shape[1] == 6, \
            f'points should have 6 channels (xyz rgb), but got {points.shape[1]}'
        if self.color_mean is not None:
            points[:, 3:] = points[:, 3:] - \
                           points[:, 3:].new_tensor(self.color_mean)
        if self.color_std is not None:
            points[:, 3:] = points[:, 3:] / \
                points[:, 3:].new_tensor(self.color_std)
        return points, targets


class ElasticTransfrom(object):
    """Apply elastic augmentation to a 3D scene. Required Keys:

    Args:
        gran (List[float]): Size of the noise grid (in same scale[m/cm]
            as the voxel grid).
        mag (List[float]): Noise multiplier.
        voxel_size (float): Voxel size.
        p (float): probability of applying this transform.
    """

    def __init__(self, gran, mag, voxel_size, p=1.0):
        self.gran = gran
        self.mag = mag
        self.voxel_size = voxel_size
        self.p = p

    def __call__(self, points, targets):
        """Private function-wrapper for elastic transform.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Results after elastic, 'points' is updated
            in the result dict.
        """
        coords = points[:, :3].numpy() / self.voxel_size
        if "query2d_pos" in targets["extra_features"]:
            # coords_query_3dctr = input_dict['query_3dctr'].tensor[:, :3].numpy() / self.voxel_size
            query2d_coords = targets["extra_features"]["query2d_pos"].numpy() / self.voxel_size
        if np.random.rand() < self.p:
            coords, noise_interp_1 = self.elastic(coords, self.gran[0], self.mag[0])
            coords, noise_interp_2 = self.elastic(coords, self.gran[1], self.mag[1])
            if "query2d_pos" in targets["extra_features"]:
                query2d_coords, _ = self.elastic(query2d_coords, self.gran[0], self.mag[0], noise_interp_1)
                query2d_coords, _ = self.elastic(query2d_coords, self.gran[1], self.mag[1], noise_interp_2)
        targets['elastic_coords'] = torch.FloatTensor(coords)
        if "query2d_pos" in targets["extra_features"]:
            targets["extra_features"]['elastic_coords_query2d_pos'] = torch.tensor(query2d_coords)
        targets["coords_voxel_size"] = self.voxel_size
        return points, targets

    def elastic(self, x, gran, mag, interp=None):
        """Private function for elastic transform to a points.

        Args:
            x (ndarray): Point cloud.
            gran (List[float]): Size of the noise grid (in same scale[m/cm]
                as the voxel grid).
            mag: (List[float]): Noise multiplier.
        
        Returns:
            dict: Results after elastic, 'points' is updated
                in the result dict.
        """
        if interp is None:
            blur0 = np.ones((3, 1, 1)).astype('float32') / 3
            blur1 = np.ones((1, 3, 1)).astype('float32') / 3
            blur2 = np.ones((1, 1, 3)).astype('float32') / 3

            noise_dim = np.abs(x).max(0).astype(np.int32) // gran + 3
            noise = [
                np.random.randn(noise_dim[0], noise_dim[1],
                                noise_dim[2]).astype('float32') for _ in range(3)
            ]

            for blur in [blur0, blur1, blur2, blur0, blur1, blur2]:
                noise = [
                    scipy.ndimage.filters.convolve(
                        n, blur, mode='constant', cval=0) for n in noise
                ]

            ax = [
                np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in noise_dim
            ]
            interp = [
                scipy.interpolate.RegularGridInterpolator(
                    ax, n, bounds_error=0, fill_value=0) for n in noise
            ]

        return x + np.hstack([i(x)[:, None] for i in interp]) * mag, interp
