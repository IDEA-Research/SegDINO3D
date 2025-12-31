# SegDINO3D
# url: https://github.com/IDEA-Research/SegDINO3D
# Copyright (c) 2025 IDEA. All Rights Reserved.

from typing import Dict, List
from segdino3d.gtypes import GD3DTarget
from mmdet3d.structures import PointData
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

from segdino3d import (
    ARCHITECTURES,
    build_backbone,
    build_decoder,
    build_loss,
    build_text_encoder,
)


def mask_matrix_nms(masks,
                    labels,
                    scores,
                    filter_thr=-1,
                    nms_pre=-1,
                    max_num=-1,
                    kernel='gaussian',
                    sigma=2.0,
                    mask_area=None):
    """Matrix NMS for multi-class masks.

    Args:
        masks (Tensor): Has shape (num_instances, m)
        labels (Tensor): Labels of corresponding masks,
            has shape (num_instances,).
        scores (Tensor): Mask scores of corresponding masks,
            has shape (num_instances).
        filter_thr (float): Score threshold to filter the masks
            after matrix nms. Default: -1, which means do not
            use filter_thr.
        nms_pre (int): The max number of instances to do the matrix nms.
            Default: -1, which means do not use nms_pre.
        max_num (int, optional): If there are more than max_num masks after
            matrix, only top max_num will be kept. Default: -1, which means
            do not use max_num.
        kernel (str): 'linear' or 'gaussian'.
        sigma (float): std in gaussian method.
        mask_area (Tensor): The sum of seg_masks.

    Returns:
        tuple(Tensor): Processed mask results.

            - scores (Tensor): Updated scores, has shape (n,).
            - labels (Tensor): Remained labels, has shape (n,).
            - masks (Tensor): Remained masks, has shape (n, m).
            - keep_inds (Tensor): The indices number of
                the remaining mask in the input mask, has shape (n,).
    """
    sort_inds_record = torch.arange(len(scores), device=scores.device)
    assert len(labels) == len(masks) == len(scores)
    if len(labels) == 0:
        return scores.new_zeros(0), labels.new_zeros(0), masks.new_zeros(
            0, *masks.shape[-1:]), labels.new_zeros(0)
    if mask_area is None:
        mask_area = masks.sum(1).float()
    else:
        assert len(masks) == len(mask_area)

    # sort and keep top nms_pre
    scores, sort_inds = torch.sort(scores, descending=True)
    sort_inds_record = sort_inds_record[sort_inds]

    keep_inds = sort_inds
    if nms_pre > 0 and len(sort_inds) > nms_pre:
        sort_inds = sort_inds[:nms_pre]
        keep_inds = keep_inds[:nms_pre]
        scores = scores[:nms_pre]
        sort_inds_record = sort_inds_record[:nms_pre]
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num_masks = len(labels)
    flatten_masks = masks.reshape(num_masks, -1).float()
    # inter.
    inter_matrix = torch.mm(flatten_masks, flatten_masks.transpose(1, 0))
    expanded_mask_area = mask_area.expand(num_masks, num_masks)
    # Upper triangle iou matrix.
    iou_matrix = (inter_matrix /
                  (expanded_mask_area + expanded_mask_area.transpose(1, 0) -
                   inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    expanded_labels = labels.expand(num_masks, num_masks)
    # Upper triangle label matrix.
    label_matrix = (expanded_labels == expanded_labels.transpose(
        1, 0)).triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(num_masks,
                                           num_masks).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # Calculate the decay_coefficient
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError(
            f'{kernel} kernel is not supported in matrix nms!')
    # update the score.
    scores = scores * decay_coefficient

    if filter_thr > 0:
        keep = scores >= filter_thr
        keep_inds = keep_inds[keep]
        if not keep.any():
            return scores.new_zeros(0), labels.new_zeros(0), masks.new_zeros(
                0, *masks.shape[-1:]), labels.new_zeros(0)
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep]
        sort_inds_record = sort_inds_record[keep]

    # sort and keep top max_num
    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = keep_inds[sort_inds]
    if max_num > 0 and len(sort_inds) > max_num:
        raise NotImplementedError("Not consider centers and sizes here.")
    masks = masks[sort_inds]
    labels = labels[sort_inds]
    sort_inds_record = sort_inds_record[sort_inds]

    return scores, labels, masks, keep_inds, sort_inds_record


@ARCHITECTURES.register_module()
class Baseline3D(nn.Module):
    def __init__(self,
            # main configs
            num_classes: int,
            pointcloud_backbone_cfg: Dict,
            decoder_cfg: Dict = None,
            criterion_cfg: Dict = None,
            text_encoder_cfg: Dict = None,
            use_sim_classifier: bool = False,
            # detailed configs
            query_thr: float = 0.5,
            test_cfg = None,
            add_positional_embedding = False,
            mode_3d_center: str = "mean",
            query_num = -1,
            filter_outofbox_points_eval: bool = False):
        super().__init__()
        # build main components
        self.backbone = build_backbone(pointcloud_backbone_cfg)
        self.decoder = build_decoder(decoder_cfg)
        self.criterion = build_loss(criterion_cfg)
        if not text_encoder_cfg is None:
            self.text_encoder = build_text_encoder(text_encoder_cfg)
        self.use_sim_classifier = use_sim_classifier
        if self.use_sim_classifier:
            assert text_encoder_cfg is not None, "Text encoder must be provided when using sim classifier."

        self.query_thr = query_thr
        self.num_classes = num_classes
        self.test_cfg = test_cfg
        self.add_positional_embedding = add_positional_embedding
        self.mode_3d_center = mode_3d_center
        self.query_num = query_num
        self.filter_outofbox_points_eval = filter_outofbox_points_eval
        if self.filter_outofbox_points_eval:
            assert self.decoder.add_box_size_pred, "When filter_outofbox_points_eval is True, decoder must have add_box_size_pred set to True."
    
    def forward_backbone(self, samples, targets):
        """
        Args:
            Same as forward.
        Outputs:
            sp_features_3d: The superpoint features after the 3D backbone. 
                List of FloatTensor. features[i]: [n x C], n superpoint features with C channels.
            sp_pos_wo_elastic: The superpoints' 3D position without affected by elastic augmentation.
                List of FloatTensor. features[i]: [n x 3], n superpoints' 3D positions (xyz).
        """
        sp_features_3d, sp_pos, sp_pos_wo_elastic = self.backbone.forward_wrapper(samples, targets, return_sp_mean_pos=True)
        return sp_features_3d, sp_pos, sp_pos_wo_elastic

    def forward_decoder(self, 
        sp_features_3d, sp_pos, sp_pos_wo_elastic, queries, queries_pos, targets, scene_range):

        if self.decoder.add_dinox_query_ca:  # Obtain the queries from 2D models.
            query2d_feat = [target["extra_features"]["query2d_feats"] for target in targets]
            query2d_pos = [target["extra_features"]["query2d_pos"] for target in targets]
        else:
            query2d_feat = query2d_pos = None
        outputs = self.decoder(
            sp_features_3d, sp_pos, sp_pos_wo_elastic, queries, queries_pos, query2d_feat, query2d_pos, scene_range)
        return outputs
    
    def _select_queries(self, x, x_pos=None, targets=None):
        """Select queries. If evaluation, return all super points as queries.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_suppoints_i, n_channels).
            x_pos (List[Tensor]): of len batch_size, each of shape
                (n_suppoints_i, 3). the 3D coordinates of superpoints.
            targets (List[InstanceData_]): of len batch_size.
                Ground truth which can contain `labels` of shape (n_gts_i,),
                `sp_inst_sem_masks` of shape (n_gts_i, n_points_i).

        Returns:
            Tuple:
                queries: List[Tensor]: Queries of len batch_size, each queries of shape
                    (n_queries_i, n_channels).
                    (n_queries_i, 3).
                targets: List[Dict]: of len batch_size, each updated
                    with `query_inst_sem_masks` of shape (n_gts_i, n_queries_i).
        """
        if not self.training and (self.query_num == -1):  # during evaluation all super points are treated as queries.
            return x, x_pos, targets
        queries = []
        queries_pos = [] if self.add_positional_embedding else None
        if self.query_num > 0:
            for i in range(len(x)):
                sp_query = self.decoder.query_proj(x[i])    # n_sp, d_model
                sp_norm_query = self.decoder.out_norm(sp_query)     # n_sp, d_model
                sp_cls_preds = self.decoder.out_cls(sp_norm_query)  # n_sp, num_classes + 1
                sp_scores = F.softmax(sp_cls_preds, dim=-1)[:, :-1] # n_sp, num_classes

                sp_scores = sp_scores.max(dim=1)[0]
                if self.query_num > 0 and sp_scores.shape[0] > self.query_num:
                    ids = torch.topk(sp_scores, self.query_num, largest=True)[1]
                else:
                    ids = torch.arange(sp_scores.shape[0], device=sp_scores.device)

                queries.append(x[i][ids])
                if targets is not None:
                    targets[i].query_inst_sem_masks = targets[i].sp_inst_sem_masks[:, ids]
                if x_pos is not None:
                    queries_pos.append(x_pos[i][ids])
            return queries, queries_pos, targets
        for i in range(len(x)):
            if self.query_thr < 1:
                n = (1 - self.query_thr) * torch.rand(1) + self.query_thr
                n = (n * len(x[i])).int()
                ids = torch.randperm(len(x[i]))[:n].to(x[i].device)
                queries.append(x[i][ids])  # random sample some super points as query
                targets[i].query_inst_sem_masks = targets[i].sp_inst_sem_masks[:, ids]  # bool, [N_instance, N_sup] -> [N_instance, n_sup]. Query Mask: each row describes the mask for an instance, describing which super points belongs to the instance.
                if x_pos is not None:
                    queries_pos.append(x_pos[i][ids])
            else:
                queries.append(x[i])
                targets[i].query_inst_sem_masks = targets[i].sp_inst_sem_masks
                if x_pos is not None:
                    queries_pos.append(x_pos[i])
        return queries, queries_pos, targets
    
    def get_extra_instance_data(self, samples, targets, add_instance_centers=False, add_instance_axis_aligned_box=False):
        """Get the center of each instance in the batch.

        Args:
            samples (Dict): Batch inputs.
            targets (List[DataSample]): Batch data samples.

        Returns:
            List[Tensor]: List of tensors containing the center of each instance.
        """
        if not (add_instance_centers or add_instance_axis_aligned_box):
            return None
        scene_range = []
        for i in range(len(targets)):
            if 'elastic_coords' in targets[i]:
                coords = targets[i]['elastic_coords'] * self.backbone.voxel_size    # (n_points, 3)
            else:
                coords = samples[i][:, :3]    # (n_points, 3)

            min_coords = coords.min(dim=0)[0]  # (3,)
            max_coords = coords.max(dim=0)[0]  # (3,)
            scene_range.append((min_coords, max_coords))

            instance_masks = targets[i]['masks'][..., 0] # (n_instance, n_points), bool mask
            instance_num = instance_masks.shape[0]
            instance_centers = torch.zeros((instance_num, 3), device=coords.device)    # (n_instances, 3)
            instance_sizes = torch.zeros((instance_num, 3), device=coords.device)  # (n_instances, 3)
            for instance_id in range(instance_num):
                instance_mask = instance_masks[instance_id] # (n_points, )
                instance_points = coords[instance_mask]   # (n_points_in_instance, 3)
                if instance_points.shape[0] > 0:
                    if self.mode_3d_center == "mean":
                        instance_centers[instance_id] = instance_points.mean(dim=0)
                    elif self.mode_3d_center == "median":
                        instance_centers[instance_id] = (instance_points.max(dim=0)[0] + instance_points.min(dim=0)[0]) / 2
                    instance_sizes[instance_id] = instance_points.max(dim=0)[0] - instance_points.min(dim=0)[0]
            if add_instance_centers:
                targets[i].instance_centers = instance_centers
            if add_instance_axis_aligned_box:
                targets[i].instance_sizes = instance_sizes
        return scene_range

    def forward(self, 
        samples: None,
        targets: List[GD3DTarget] = None,):
        """
        Args:
            samples: List of FloatTensor. Samples[i]: [N x 6], the point cloud of the scene with
                N points, 6 channels indicate the xyz(0-2), rgb(3-5).
            targets: Lift of Dict. Annotations and extra features (including the features from 2D Foundation model).
        """
        scene_range = self.get_extra_instance_data(samples, targets, self.add_positional_embedding, self.decoder.add_box_size_pred)
        sp_features_3d, sp_pos, sp_pos_wo_elastic = self.forward_backbone(samples, targets)
        queries, queries_pos, targets = self._select_queries(sp_features_3d, sp_pos, targets)
        try:
            self.decoder.return_hidden_states = False if self.training else True
            self.decoder.return_aux_outputs = True
        except Exception:
            pass
        outputs = self.forward_decoder(sp_features_3d, sp_pos, sp_pos_wo_elastic, queries, queries_pos, targets, scene_range)
        try:
            del sp_features_3d
            del sp_pos
            del sp_pos_wo_elastic
            torch.cuda.empty_cache()
        except Exception:
            pass
        if not self.training:
            with torch.no_grad():
                pred_pts_seg = self.predict_by_feat(  # only support bs=1
                    samples, outputs, targets[0]["extra_features"]["super_point_masks"])
            targets[0].pred_pts_seg = pred_pts_seg[0]
            return targets
        else:
            try:
                if isinstance(outputs, dict) and 'hidden_states' in outputs:
                    del outputs['hidden_states']
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return self.criterion(outputs, targets)
    
    def filter_outofbox_points(self, points, mask_pred, center_pred, size_pred, loose_ratio = 1.5):
        """
        Args:
            points: N X 3
            mask_pred: N_obj X N
            center_pred: N_obj X 3
            size_pred: N_obj X 3
        """
        filtered_masks = []
        for mask, center, size in zip(mask_pred, center_pred, size_pred):
            size = size * (1 + loose_ratio)  # enlarge the box size
            bbox = [center - size / 2, center + size / 2]  # [min, max]
            mask_in_box = (
                (points[:, 0] >= bbox[0][0]) & (points[:, 0] <= bbox[1][0]) &
                (points[:, 1] >= bbox[0][1]) & (points[:, 1] <= bbox[1][1]) &
                (points[:, 2] >= bbox[0][2]) & (points[:, 2] <= bbox[1][2])
            )
            mask[~mask_in_box] = False
            filtered_masks.append(mask)
        if len(filtered_masks) > 0:
            filtered_masks = torch.stack(filtered_masks, dim=0)  # N_obj X N
        else:
            filtered_masks = mask_pred.new_zeros((0, mask_pred.shape[1]), dtype=torch.bool)
        return filtered_masks

    def predict_by_feat(self, samples, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        inst_res = self.predict_by_feat_instance(
            samples, out, superpoints, self.test_cfg.inst_score_thr)
        sem_res = self.predict_by_feat_semantic(out, superpoints)
        pan_res = self.predict_by_feat_panoptic(samples, out, superpoints)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(),
                             pan_res[1].cpu().numpy()]
      
        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].detach().cpu().numpy(),
                instance_scores=inst_res[2].detach().cpu().numpy(),
                sort_and_mask=inst_res[3],
                instance_boxes=inst_res[4].detach().cpu().numpy() if inst_res[4] is not None else np.zeros((inst_res[2].shape[0], 6)))]
    
    def predict_by_feat_instance(self, samples, out, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        cls_preds = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        points = samples[0][:, :3]

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        if self.test_cfg.get('obj_normalization', None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _, sort_inds_record = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred_sigmoid = mask_pred_sigmoid[:, superpoints]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]
        sort_inds_record = sort_inds_record[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]
        sort_inds_record = sort_inds_record[npoint_mask]
        if "centers" in out and out["centers"][0] is not None:
            center_pred = out["centers"][0][topk_idx][sort_inds_record]
        else:
            center_pred = None
        if "sizes" in out and out["sizes"][0] is not None:
            size_pred = out["sizes"][0][topk_idx][sort_inds_record]
        else:
            size_pred = None
        if center_pred is not None and size_pred is not None:
            box_pred = torch.cat([center_pred, size_pred], dim=-1)
        else:
            box_pred = None
        if self.filter_outofbox_points_eval:
            mask_pred = self.filter_outofbox_points(points, mask_pred, center_pred, size_pred)

        sort_and_mask = (topk_idx, score_mask, npoint_mask)
        return mask_pred, labels, scores, sort_and_mask, box_pred

    def predict_by_feat_semantic(self, out, superpoints, classes=None):
        """Predict semantic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `sem_preds` of shape (n_queries, n_semantic_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            classes (List[int] or None): semantic (stuff) class ids.
        
        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, n_semantic_classe + 1),
        """
        if classes is None:
            classes = list(range(out['sem_preds'][0].shape[1] - 1))
        if self.query_num == -1:
            return out['sem_preds'][0][:, classes].argmax(dim=1)[superpoints]
        else:
            fake_superpoints = torch.zeros_like(superpoints)
            return out['sem_preds'][0][:, classes].argmax(dim=1)[fake_superpoints]

    def predict_by_feat_panoptic(self, samples, out, superpoints):
        """Predict panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        sem_map = self.predict_by_feat_semantic(
            out, superpoints, self.test_cfg.stuff_classes)
        mask_pred, labels, scores, _, box_pred  = self.predict_by_feat_instance(
            samples, out, superpoints, self.test_cfg.pan_score_thr)
        if mask_pred.shape[0] == 0:
            return sem_map, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        n_stuff_classes = len(self.test_cfg.stuff_classes)
        inst_idxs = torch.arange(
            n_stuff_classes, 
            mask_pred.shape[0] + n_stuff_classes, 
            device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs] + n_stuff_classes

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map[things_inst_mask != 0] = 0
        inst_map = sem_map.clone()
        inst_map += things_inst_mask
        sem_map += things_sem_mask
        return sem_map, inst_map
