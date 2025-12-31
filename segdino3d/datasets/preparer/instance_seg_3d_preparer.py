import torch
import torch.utils.data


from typing import Dict
from segdino3d import PREPARERS


@PREPARERS.register_module()
class InstanceSeg3DDataPreparer(object):
    def __init__(self,
                 min_area:int=0):
        self.min_area = min_area

    def __call__(self, results: Dict):
        points = results['points']
        N = points.shape[0]
        if "scene_id" in results:
            scene_id = results['scene_id']
        else:
            scene_id = None
        anno = results['target']
        # filter crowded boxes
        anno = [
            obj for obj in anno if ('iscrowd' not in obj or obj['iscrowd'] == 0) and obj['area'] > self.min_area
        ]

        # prepare classess
        classes = [obj["category_id"] for obj in anno]
        # classes = [1 for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        masks = torch.stack([obj["instance_mask"] for obj in anno], dim=0)
        target["masks"] = masks
        target["labels"] = classes
        if "sp_inst_sem_masks" in results["extra_features"]:
            target["sp_inst_sem_masks"] = results["extra_features"].pop("sp_inst_sem_masks").T  # num_instance + num_classes + 1, num_super_points
        if scene_id is not None:
            target["scene_id"] = torch.tensor([scene_id])

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        target["orig_size"] = torch.as_tensor([int(N)])
        target["size"] = torch.as_tensor([int(N)])

        target["extra_features"] = results["extra_features"]
        return dict(points=points, target=target)
