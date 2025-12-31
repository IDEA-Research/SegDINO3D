import os
import numpy as np
from typing import Dict, Union

import PIL.Image as Image
from PIL import ImageFile
import torch.nn.functional as F
from torch_scatter import scatter_mean

ImageFile.LOAD_TRUNCATED_IMAGES = True  # support loading truncated image
import torch
from torch.utils.data import Dataset
from segdino3d import DATASETS, build_transform
from segdino3d.gtypes import GD3DTarget


@DATASETS.register_module()
class ScanNet200InstanceSeg3D(Dataset):
    """
    ScanNet200 dataset for close set 3D instance segmentation.
    We follow Oneformer3d's preprocess of the origin scannet200 data.
    Args:
        split (str): train, val, test
        root_scenes (str): Root directory of the ScanNet200 dataset.
        use_super_points (bool): Whether to cluster points into super points. 
            You can refer to the oneformer3D's preprocess: oneformer3d/data/scannet/batch_load_scannet_data.py
    Returns:
        dict: A dictionary containing the following keys:
            - scene_id: Scene ID. Shape = (1,)
    """

    def __init__(self, 
                 scene_set,
                 root_scenes,
                 use_super_points=False,
                 adjust_class_ids=True,
                 exclude_stuffs=True,
                 root_points_2dfeats=None,
                 dropout_rate_2dfeats=0.0,
                 transform_cfg=None,
                 mode_fuse_multi_scale_2d_feats="mean",
                 stuff_categories=["wall", "floor"],
                 dataset_type = "scannet200_InstanceSeg3D",
                 loss_branch = "cdn",
        ):
        super().__init__()
        self.scene_set = scene_set
        assert scene_set in ["train", "val", "test"], f"Invalid scene set: {scene_set}"
        self.root_scenes = root_scenes
        self.root_points_2dfeats = root_points_2dfeats
        self.dropout_rate_2dfeats = dropout_rate_2dfeats
        self.use_super_points = use_super_points
        self.scene_ids = self.get_all_scene_ids()
        # for class adjustment.
        self.stuff_categories = stuff_categories
        self.exclude_stuffs = exclude_stuffs  # whether to exclude stuffs in instance segmentation / semantic segmentation.
        self.category_dict, self.category_id_list, \
            self.category_list_excluded, self.category_id_list_excluded, \
            self.stuff_category_id_list = self.get_category_dict()
        self.bg_class_id = 200
        self.adjust_class_ids = adjust_class_ids
        if self.adjust_class_ids:
            self.seg_label_mapping = np.load("scannet200_seg_label_mapping.npy", allow_pickle=True)  # mapping classes from 1~1191 to 0-199 of all fg objects (including stuffs)
        # transforms
        transform_cfg.update(
            scene_set=scene_set, 
            category_dict=self.category_dict
        )
        self._transforms = build_transform(transform_cfg)
        self.dataset_type = dataset_type
        self.loss_branch = loss_branch

        # for 2D features from vision foundation model.
        self.mode_fuse_multi_scale_2d_feats = mode_fuse_multi_scale_2d_feats
    
    def get_category_dict(self):
        """
        Get the category dictionary for the dataset. Class IDs is set to 0-199.
        """
        self.category_list = ['wall', 'floor', 'chair', 'table', 'door', 'couch', 'cabinet',
                'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink',
                'picture', 'window', 'toilet', 'bookshelf', 'monitor',
                'curtain', 'book', 'armchair', 'coffee table', 'box',
                'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes',
                'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion',
                'plant', 'ceiling', 'bathtub', 'end table', 'dining table',
                'keyboard', 'bag', 'backpack', 'toilet paper', 'printer',
                'tv stand', 'whiteboard', 'blanket', 'shower curtain',
                'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe',
                'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board',
                'washing machine', 'mirror', 'copier', 'basket', 'sofa chair',
                'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person',
                'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate',
                'blackboard', 'piano', 'suitcase', 'rail', 'radiator',
                'recycling bin', 'container', 'wardrobe', 'soap dispenser',
                'telephone', 'bucket', 'clock', 'stand', 'light',
                'laundry basket', 'pipe', 'clothes dryer', 'guitar',
                'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle',
                'ladder', 'bathroom stall', 'shower wall', 'cup', 'jacket',
                'storage bin', 'coffee maker', 'dishwasher',
                'paper towel roll', 'machine', 'mat', 'windowsill', 'bar',
                'toaster', 'bulletin board', 'ironing board', 'fireplace',
                'soap dish', 'kitchen counter', 'doorframe',
                'toilet paper dispenser', 'mini fridge', 'fire extinguisher',
                'ball', 'hat', 'shower curtain rod', 'water cooler',
                'paper cutter', 'tray', 'shower door', 'pillar', 'ledge',
                'toaster oven', 'mouse', 'toilet seat cover dispenser',
                'furniture', 'cart', 'storage container', 'scale',
                'tissue box', 'light switch', 'crate', 'power outlet',
                'decoration', 'sign', 'projector', 'closet door',
                'vacuum cleaner', 'candle', 'plunger', 'stuffed animal',
                'headphones', 'dish rack', 'broom', 'guitar case',
                'range hood', 'dustpan', 'hair dryer', 'water bottle',
                'handicap bar', 'purse', 'vent', 'shower floor',
                'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock',
                'music stand', 'projector screen', 'divider',
                'laundry detergent', 'bathroom counter', 'object',
                'bathroom vanity', 'closet wall', 'laundry hamper',
                'bathroom stall door', 'ceiling light', 'trash bin',
                'dumbbell', 'stair rail', 'tube', 'bathroom cabinet',
                'cd case', 'closet rod', 'coffee kettle', 'structure',
                'shower head', 'keyboard piano', 'case of water bottles',
                'coat rack', 'storage organizer', 'folded chair', 'fire alarm',
                'power strip', 'calendar', 'poster', 'potted plant', 'luggage',
                'mattress']
        category_id_list = [i for i in range(0, 200)]
        categories = [{"id": cat_id, "name": cat_name} for cat_name, cat_id in zip(self.category_list, category_id_list)]
        excluded_classes = []
        excluded_class_ids = [cat_id for cat_name, cat_id in zip(self.category_list, category_id_list) if cat_name in excluded_classes]
        stuff_category_id_list = [cat_id for cat_name, cat_id in zip(self.category_list, category_id_list) if cat_name in self.stuff_categories]
        if self.exclude_stuffs:
            categories_ = []
            for cat in categories:
                if cat["name"] not in self.stuff_categories:
                    categories_.append({
                        "id": cat["id"] - len(self.stuff_categories),  # reassign the id to 0-198.
                        "name": cat["name"]
                    })
            categories = categories_
        return categories, category_id_list, excluded_classes, excluded_class_ids, stuff_category_id_list

    def get_all_scene_ids(self):
        """
        Get all scene ids in the scene set (train / val).
        Returns:
            list: A list of scene ids.
        """
        path_scene_set = os.path.join(self.root_scenes, "meta_data", f"scannetv2_{self.scene_set}.txt")
        scene_list = []
        with open(path_scene_set, "r") as f:
            for line in f.readlines():
                scene_list.append(line.strip())
        return scene_list

    def adjust_class_ids_(self, semantic_masks):
        """
        Swap between chair and floor. (A bug in the scannet200 dataset.)
        mapping classes from 1~1191 to 0-199 of all fg objects (including stuffs)
        """
        # 1. SwapChairAndFloor. 
        mask = semantic_masks.copy()
        mask[semantic_masks == 2] = 3
        mask[semantic_masks == 3] = 2
        semantic_masks = mask

        # 2. PointSegClassMapping
        semantic_masks = self.seg_label_mapping[semantic_masks]
        return semantic_masks
    
    def exclude_stuffs_(self, instance_masks, semantic_masks):
        """
        Exclude stuffs from instance mask.
        Set points belong to stuff instances to -1, set the background instance to -1. Reassign instance ids.
        """
        for cls_id in self.stuff_category_id_list:
            instance_masks[semantic_masks == cls_id] = -1
        instance_masks[semantic_masks == len(self.category_id_list)] = -1
        instance_ids = np.unique(instance_masks)
        new_instance_ids = np.arange(len(instance_ids)) - 1
        instance_id_mapping = np.zeros(instance_masks.max() + 2)
        instance_id_mapping[instance_ids] = new_instance_ids
        instance_masks = instance_id_mapping[instance_masks]
        return instance_masks

    def merge_stuffs_(self, instance_masks, semantic_masks):
        """
        Merge stuff objects to one object for each stuff classes, and add the stuff instances to the instance masks.
        """
        if not self.scene_set == "train":
            instance_masks[instance_masks!=-1] += len(self.stuff_category_id_list)  # +2 for non-stuff & bg points.
            for idx, stuff_id in enumerate(self.stuff_category_id_list):  # merge each stuff instances into one instance.
                instance_masks[semantic_masks == stuff_id] = idx  
        return instance_masks

    def __len__(self):
        return len(self.scene_ids)
    
    def __getitem__(self, idx_scale):
        if isinstance(idx_scale, tuple):
            # this for sync scale
            idx, current_scale = idx_scale
        else:
            idx = idx_scale
            current_scale = None
        
        scene_id = self.scene_ids[idx]
        path_scene_points = os.path.join(self.root_scenes, "points", f"{scene_id}.bin")
        points = np.fromfile(path_scene_points, dtype=np.float32).reshape(-1, 6)
        # load ground truth
        path_instance_gt = os.path.join(self.root_scenes, "instance_mask", f"{scene_id}.bin")
        path_semantic_gt = os.path.join(self.root_scenes, "semantic_mask", f"{scene_id}.bin")
        instance_masks = np.fromfile(path_instance_gt, dtype=np.int64).reshape(-1, 1)
        semantic_masks = np.fromfile(path_semantic_gt, dtype=np.int64).reshape(-1, 1)
        if self.adjust_class_ids:  # project class id to 0-199.
            semantic_masks = self.adjust_class_ids_(semantic_masks)
        if self.exclude_stuffs:  # in instance segmentation segmentation, we need to exclude stuffs from instances.
            instance_masks = self.exclude_stuffs_(instance_masks, semantic_masks)
        # load 2D features sampled from 2D image feature maps.
        if not self.root_points_2dfeats is None:
            path_points_2dfeats = os.path.join(self.root_points_2dfeats, f"{scene_id}.pth")
            path_query2d_feats  = os.path.join(self.root_points_2dfeats, f"{scene_id}_query_feats.pth")
            path_query2d_pos    = os.path.join(self.root_points_2dfeats, f"{scene_id}_query_3dctr.pth")

            points_2dfeats = torch.load(path_points_2dfeats)
            query2d_feats = torch.load(path_query2d_feats)
            query2d_pos = torch.load(path_query2d_pos)
            if self.dropout_rate_2dfeats > 0.0:
                num_query = query2d_pos.shape[0]
                num_sample = int(num_query * (1 - self.dropout_rate_2dfeats))
                sample_idx = np.random.choice(num_query, num_sample, replace=False)
                query2d_pos = query2d_pos[sample_idx]
                query2d_feats = query2d_feats[sample_idx]
            if self.mode_fuse_multi_scale_2d_feats == "mean":
                points_2dfeats = torch.stack(points_2dfeats, dim=0).mean(dim=0)
            else:
                raise NotImplementedError
        else:
            points_2dfeats = query2d_feats = query2d_pos = None
        # load super point masks.
        if self.use_super_points:
            path_super_points = os.path.join(self.root_scenes, "super_points", f"{scene_id}.bin")
            super_point_masks = np.fromfile(path_super_points, dtype=np.int64)
            instance_masks_ = torch.LongTensor(instance_masks)
            instance_masks_[instance_masks_ == -1] = int(instance_masks_.max() + 1)  # set -1 to max+1, so that we can use one-hot encoding.
            instance_masks_onehot = F.one_hot(instance_masks_.squeeze(-1))[:, :-1]  # remove the last channel, which is the background.
            sp_inst_masks = scatter_mean(instance_masks_onehot.float(), torch.tensor(super_point_masks), dim=0)
            sp_inst_masks = sp_inst_masks > 0.5  # if more than half of the super point belongs to the instance, we consider it as the instance. There will be at most 1 be true.
            # sp_inst_masks = sp_inst_masks.float().argmax(dim=-1)  # we assume the one super point only belongs to one instance.
            semantic_masks_ = F.one_hot(torch.LongTensor(semantic_masks).squeeze(-1), num_classes=len(self.category_id_list) + 1)  # last channel is the background.
            sp_sem_masks = scatter_mean(semantic_masks_.float(), torch.tensor(super_point_masks), dim=0)
            sp_sem_masks = sp_sem_masks > 0.5
            sp_sem_masks[sp_sem_masks.sum(dim=-1) == 0, -1] = True
            sp_inst_sem_masks = torch.cat([sp_inst_masks, sp_sem_masks], dim=-1)  # num_super_points, num_instance + (num_classes + 1)
        else:
            super_point_masks = None

        if self.scene_set != "train":
            # This is necessary for the evaluation.
            # during the evaluation of instance segmentation, we need to merge stuffes into one instance.
            instance_masks = self.merge_stuffs_(instance_masks, semantic_masks)
        target_list = self.split_instance_gt(
            torch.tensor(instance_masks),
            torch.tensor(semantic_masks),
            torch.tensor(super_point_masks),
            torch.tensor(sp_inst_masks),
        )
        results = {
            "scene_id": idx,
            "points": torch.tensor(points),
            "extra_features": {
                "points_2dfeats": points_2dfeats,
                "query2d_feats": query2d_feats,
                "query2d_pos": query2d_pos,
                "super_point_masks": torch.tensor(super_point_masks),
                "sp_inst_sem_masks": sp_inst_sem_masks,  # target of super points' semantic segmentation.
                "class_name_list": self.category_list[len(self.stuff_category_id_list):] + ["background"],  # class names of the instance segmentation.
            },
            "target": target_list,  # targets of instance segmentation.
        }
        results = self._transforms(results)

        points = results["points"]
        target = results["target"]
        target["scene_id"] = scene_id
        target["data_source"] = self.dataset_type + ":" + str(idx)
        target["loss_branch"] = self.loss_branch
        target["prompt_type"] = "text"

        return points, GD3DTarget(**target)

    def split_instance_gt(self, instance_masks, semantic_masks, super_point_masks, sp_inst_masks):
        """
        Split the scene-level gt into instance-level gt.
        Args:
            instance_masks (torch.Tensor): Shape = (N, 1)
            semantic_masks (torch.Tensor): Shape = (N, 1)
            super_point_masks (torch.Tensor): Shape = (N, 1)
        Returns: A list of dicts.
            {
                "scene_id": scene_id,
                "instance_masks": torch.tensor(instance_masks),
                "semantic_masks": torch.tensor(semantic_masks),
            }
        """
        instance_ids = torch.unique(instance_masks)
        instance_ids = instance_ids[instance_ids >= 0]  # remove the background
        target_list = []
        for instance_id in instance_ids:
            instance_mask = (instance_masks == instance_id)
            sp_instance_mask = (sp_inst_masks == instance_id)
            category_id = semantic_masks[instance_mask]
            # assert torch.unique(category_id).shape[0] == 1, f"Instance {instance_id} has multiple categories."
            category_id = category_id[0].item()
            if category_id in self.category_id_list_excluded:
                continue
            if self.scene_set == "train":  
                category_id = category_id - len(self.stuff_category_id_list) if self.exclude_stuffs else category_id  # we need to exlude stuffs from the category id during instance semgnetation.
            target = {
                "instance_id": instance_id,
                "instance_mask": instance_mask,
                "instance_sp_mask": sp_instance_mask,
                "category_id": category_id,
                "area": instance_mask.sum(),
            }
            target_list.append(target)
        return target_list

    def visualize_one_data_beforeT(self, 
            data, 
            dir_vis="./visualize_debug",
            vis_instance_mask=True,
            vis_semantic_mask=True,
            vis_super_point_mask=True,
        ):
        """
        Visualize the datas obtained from __getitem__ before transforms.
        Visualize the instance / semantic mask and the superpoint mask
            by saving them to "dir_vis" as .ply file.
        """
        import open3d as o3d
        def save_point_cloud(points, file_path):
            """
            points: np.ndarray, shape = (N, 6)
            """
            if isinstance(points, torch.Tensor):
                points = points.cpu().numpy()  # Convert tensor to NumPy array on the CPU
            points = np.asarray(points, dtype=np.float32)  # Ensure points are in float32 format
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points[:, :3])
            if points.shape[1] == 6:
                pc.colors = o3d.utility.Vector3dVector(points[:, 3:])
            o3d.io.write_point_cloud(file_path, pc)
        
        scene_id = data["scene_id"]
        points = data["points"]
        if vis_super_point_mask:
            superpoint_mask = data["super_point_masks"]
            colors = torch.rand(superpoint_mask.max() + 1, 3)
            superpoint_colors = colors[superpoint_mask.squeeze(-1)]
            path_superpoint_mask = os.path.join(dir_vis, f"{scene_id}_superpoint_mask.ply")
            save_point_cloud(torch.cat([points[:, :3], superpoint_colors], dim=-1), path_superpoint_mask)
        if vis_instance_mask:
            instance_masks = data["instance_masks"]
            num_instance = instance_masks.max() + 1
            colors = torch.rand(num_instance, 3)
            instance_colors = colors[instance_masks.squeeze(-1)]
            path_instance_mask = os.path.join(dir_vis, f"{scene_id}_instance_mask.ply")
            save_point_cloud(torch.cat([points[:, :3], instance_colors], dim=-1), path_instance_mask)
        if vis_semantic_mask:
            classes = torch.unique(data["semantic_masks"])
            colors = torch.rand(classes.shape[0], 3)
            semantic_colors = torch.zeros_like(points[:, :3])
            for cls_id, cls in enumerate(classes):
                semantic_colors[data["semantic_masks"].squeeze() == cls] = colors[cls_id]
            path_semantic_mask = os.path.join(dir_vis, f"{scene_id}_semantic_mask.ply")
            save_point_cloud(torch.cat([points[:, :3], semantic_colors], dim=-1), path_semantic_mask)

    def visualize_one_data_afterT(self,
            data,
            dir_vis="./visualize_debug",
        ):
        """
        Visualize the coordinates of the points and the instance / semantic mask.
            If have query2d, visualize the query2d as well.
            If elastic transform is applied, visualize the transformed points as well.
        Visualize the instance / semantic mask and the superpoint mask

        """
        import open3d as o3d
        def save_point_cloud(points, file_path):
            """
            points: np.ndarray, shape = (N, 6)
            """
            if isinstance(points, torch.Tensor):
                points = points.cpu().numpy()  # Convert tensor to NumPy array on the CPU
            points = np.asarray(points, dtype=np.float32)  # Ensure points are in float32 format
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points[:, :3])
            if points.shape[1] == 6:
                pc.colors = o3d.utility.Vector3dVector(points[:, 3:])
            o3d.io.write_point_cloud(file_path, pc)
        
        # coords
        points, targets = data
        coords = points[:, :3] / targets["coords_voxel_size"]
        query2d_coords = targets["extra_features"]["query2d_pos"] / targets["coords_voxel_size"]
        if "elastic_coords" in targets:
            elastic_coords = targets["elastic_coords"]
            elastic_query2d_coords = targets["extra_features"]["elastic_coords_query2d_pos"]
        
        path_coords = os.path.join(dir_vis, f"{targets['scene_id']}_coords.ply")
        path_elastic_coords = os.path.join(dir_vis, f"{targets['scene_id']}_elastic_coords.ply")
        path_query2d_coords = os.path.join(dir_vis, f"{targets['scene_id']}_query2d_coords.ply")
        path_query2d_elastic_coords = os.path.join(dir_vis, f"{targets['scene_id']}_query2d_elastic_coords.ply")
        color_coords = torch.ones_like(coords) * 1
        color_query2d_coords = torch.ones_like(query2d_coords) * 0.0
        save_point_cloud(torch.cat([coords, color_coords], dim=-1), path_coords)
        save_point_cloud(torch.cat([elastic_coords, color_coords], dim=-1), path_elastic_coords)
        save_point_cloud(torch.cat([query2d_coords, color_query2d_coords], dim=-1), path_query2d_coords)
        save_point_cloud(torch.cat([elastic_query2d_coords, color_query2d_coords], dim=-1), path_query2d_elastic_coords)

        # instance mask
        instance_masks = targets["masks"]
        sp_instance_masks = targets["sp_masks"]
        superpoint_masks = targets["extra_features"]["super_point_masks"]
        num_instance = instance_masks.shape[0]
        colors = torch.rand(num_instance, 3)
        for instance_id, instance_color, instance_mask in zip(range(num_instance), colors, instance_masks):
            instance_mask = instance_mask.squeeze(-1)
            instance_points = points[instance_mask, :3]
            path_instance_points = os.path.join(dir_vis, f"instance_{instance_id}_point.ply")
            save_point_cloud(torch.cat([instance_points, instance_color[None, :].repeat(instance_points.shape[0], 1)], dim=-1), path_instance_points)
            instance_sp_ids = torch.where(sp_instance_masks[instance_id])[0]
            instance_sp_points = points[(superpoint_masks.unsqueeze(1) == instance_sp_ids.unsqueeze(0)).sum(dim=1)>0, :3]
            path_instance_sp_points = os.path.join(dir_vis, f"instance_{instance_id}_sppoint.ply")
            save_point_cloud(torch.cat([instance_sp_points, instance_color[None, :].repeat(instance_sp_points.shape[0], 1)], dim=-1), path_instance_sp_points)
