from typing import Dict, List

import torch
from segdino3d import (TRANSFORMS, build_preparer, build_transform)


@TRANSFORMS.register_module()
class Segment3DTransform(object):
    """
    The transform pipeline consists of three parts: preparer, and transform.
        - preparer: Convert the annotation from raw list to tensor.
        - transform: Apply data augmentation to the image.
    
    Args:
        preparer_cfg (dict): Config for preparer.
        transform_cfg (dict): Config for transform.
        image_set (str): The image set to use. Must be one of 'train'. Defaults to None. Typically,
            we will pass this parameter from the dataset.
        category_dict (dict): The category dictionary. It is used to map category id to category
            name. Defaults to None. Typically, we will pass this parameter from the dataset.
    """

    def __init__(
        self,
        preparer_cfg: Dict = None,
        transform_cfg: Dict = None,
        scene_set: str = None,
        category_dict: Dict = None,
    ) -> None:

        self.preparer = build_preparer(preparer_cfg)

        transform_cfg.update(dict(scene_set=scene_set))
        self.transform = build_transform(transform_cfg)
        self.scene_set = scene_set
        self.category_dict = category_dict

    def __call__(self, results: Dict) -> Dict:
        """Apply transform to results.

        Args:
            results (dict): The results to apply transform. It should contain the following keys:
                - image (PIL.Image): The image in PIL format.
                - target (List[Dict]): The annotations in COCO format (List of Dict).
                (Optional):
                    - image_path (str): The path to the image. (Default: None) This is used for
                        visualization during evaluation. Typically only works for COCO dataset.

        Returns:
            dict: The transformed results with keys:
                - image (torch.Tensor): The transformed image.
                - target (Dict): The transformed annotations
        """
        current_scale = results.get('current_scale', None)
        results = self.preparer(results)
        if current_scale:
            results['target']['current_scale'] = current_scale
            
        if self.transform:
            results['points'], results['target'] = self.transform(results['points'], results['target'])
        else:
            results['points'], results['target'] = results['points'], results['target']
        return results
