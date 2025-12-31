# ------------------------------------------------------------------------
# SegDINO3D
# url: https://github.com/IDEA-Research/SegDINO3D
# Copyright (c) 2025 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from: OneFormer3D (https://github.com/filaPro/oneformer3d)
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F

from segdino3d import LOSSES
from scipy.optimize import linear_sum_assignment


class InstanceData:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        """Get attribute with default value."""
        return getattr(self, key, default)

        
class ScanNetSemanticCriterion:
    """Semantic criterion for ScanNet.

    Args:
        ignore_index (int): Ignore index.
        loss_weight (float): Loss weight.
    """

    def __init__(self, ignore_index, loss_weight):
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def __call__(self, pred, insts):
        """Calculate loss.

        Args:
            pred (dict): Predictions with List `sem_preds`
                of len batch_size, each of shape
                (n_queries_i, n_classes + 1).
            insts (list): Ground truth of len batch_size, 
                each of shape (n_classes + 1, n_queries_i).

        Returns:
            Dict: with semantic loss value.
        """
        losses = []
        for pred_mask, gt_mask in zip(pred['sem_preds'], insts):
            if self.ignore_index >= 0:
                pred_mask = pred_mask[:, :-1]
            losses.append(F.cross_entropy(
                pred_mask,
                gt_mask.sp_masks.float().argmax(0),
                ignore_index=self.ignore_index))
        loss = self.loss_weight * torch.mean(torch.stack(losses))
        return dict(seg_loss=loss)


def batch_sigmoid_bce_loss(inputs, targets):
    """Sigmoid BCE loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).
    
    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction='none')

    pos_loss = torch.einsum('nc,mc->nm', pos, targets)
    neg_loss = torch.einsum('nc,mc->nm', neg, (1 - targets))
    return (pos_loss + neg_loss) / inputs.shape[1]


def batch_dice_loss(inputs, targets):
    """Dice loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).
    
    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def get_iou(inputs, targets):
    """IoU for to equal shape masks.

    Args:
        inputs (Tensor): of shape (n_gts, n_points).
        targets (Tensor): of shape (n_gts, n_points).
    
    Returns:
        Tensor: IoU of shape (n_gts,).
    """
    inputs = inputs.sigmoid()
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_loss(inputs, targets):
    """Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs.
            Stores the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
    
    Returns:
        Tensor: loss value.
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


class QueryClassificationCost:
    """Classification cost for queries.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                must contain `scores` of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `labels` of shape (n_gts,).

        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        scores = pred_instances.scores.softmax(-1)
        cost = -scores[:, gt_instances.labels]
        return cost * self.weight


class MaskBCECost:
    """Sigmoid BCE cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                must contain `masks` of shape (n_queries, n_points).
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points).
        
        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        cost = batch_sigmoid_bce_loss(
            pred_instances.masks, gt_instances.masks.float())
        return cost * self.weight


class MaskDiceCost:
    """Dice cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                must contain `masks` of shape (n_queries, n_points).
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `masks` of shape (n_gts, n_points).
        
        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        cost = batch_dice_loss(
            pred_instances.masks, gt_instances.masks.float())
        return cost * self.weight
    
class CenterL1Cost:
    """L1 cost for centers.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute center L1 cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                must contain `instance_centers` of shape (n_queries, 3).
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `instance_centers` of shape (n_gts, 3).
        
        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        if pred_instances.get('centers') is not None:
            cost = torch.cdist(pred_instances.centers, gt_instances.instance_centers[:, :3], p=1)
        else:
            n_queries = pred_instances.masks.shape[0]
            n_gts = gt_instances.masks.shape[0]
            cost = pred_instances.masks.new_zeros((n_queries, n_gts))
        return cost * self.weight

class SizeL1Cost:
    """L1 cost for centers.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute center L1 cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                must contain `instance_centers` of shape (n_queries, 3).
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `instance_centers` of shape (n_gts, 3).
        
        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        if pred_instances.get('sizes') is not None:
            cost = torch.cdist(pred_instances.sizes, gt_instances.instance_sizes[:, :3], p=1)
        else:
            n_queries = pred_instances.masks.shape[0]
            n_gts = gt_instances.masks.shape[0]
            cost = pred_instances.masks.new_zeros((n_queries, n_gts))
        return cost * self.weight

class HungarianMatcher:
    """Hungarian matcher.

    Args:
        costs (List[ConfigDict]): Cost functions.
    """
    def __init__(self, costs):
        self.costs = []
        for cost in costs:
            cost_type = cost.pop('type', None)
            self.costs.append(globals().get(cost_type)(**cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                can contain `masks` of shape (n_queries, n_points), `scores`
                of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which can contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points).

        Returns:
            Tuple:
                - Tensor: Query ids of shape (n_matched,),
                - Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))
        
        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))
        cost_value = torch.stack(cost_values).sum(dim=0)
        query_ids, object_ids = linear_sum_assignment(cost_value.cpu().numpy())
        return labels.new_tensor(query_ids), labels.new_tensor(object_ids)


class SparseMatcher:
    """Match only queries to their including objects.

    Args:
        costs (List[Callable]): Cost functions.
        topk (int): Limit topk matches per query.
    """

    def __init__(self, costs, topk):
        self.topk = topk
        self.costs = []
        self.inf = 1e8
        for cost in costs:
            cost_type = cost.pop('type', None)
            self.costs.append(globals().get(cost_type)(**cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                can contain `masks` of shape (n_queries, n_points), `scores`
                of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which can contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points),
                `query_masks` of shape (n_gts, n_queries).

        Returns:
            Tuple:
                Tensor: Query ids of shape (n_matched,),
                Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))
        
        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))
        # of shape (n_queries, n_gts)
        cost_value = torch.stack(cost_values).sum(dim=0)
        cost_value = torch.where(
            gt_instances.query_masks.T, cost_value, self.inf)

        values = torch.topk(
            cost_value, self.topk + 1, dim=0, sorted=True,
            largest=False).values[-1:, :]
        ids = torch.argwhere(cost_value < values)
        return ids[:, 0], ids[:, 1]
    

class InstanceCriterion:
    """Instance criterion.

    Args:
        matcher (Callable): Class for matching queries with gt.
        loss_weight (List[float]): 4 weights for query classification,
            mask bce, mask dice, and score losses.
        non_object_weight (float): no_object weight for query classification.
        num_classes (int): number of classes.
        fix_dice_loss_weight (bool): Whether to fix dice loss for
            batch_size != 4.
        iter_matcher (bool): Whether to use separate matcher for
            each decoder layer.
        fix_mean_loss (bool): Whether to use .mean() instead of .sum()
            for mask losses.

    """

    def __init__(self, matcher, loss_weight, non_object_weight, num_classes,
                 fix_dice_loss_weight, iter_matcher, fix_mean_loss=False):
        type = matcher.pop('type', None)
        assert (type == 'SparseMatcher') or (type == 'HungarianMatcher'), \
            "Matcher type must be 'SparseMatcher' or 'HungarianMatcher'."
        if type == 'HungarianMatcher':
            self.matcher = HungarianMatcher(**matcher)
        elif type == 'SparseMatcher':
            self.matcher = SparseMatcher(**matcher)
        class_weight = [1] * num_classes + [non_object_weight]
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.iter_matcher = iter_matcher
        self.fix_mean_loss = fix_mean_loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_layer_loss(self, aux_outputs, insts, indices=None):
        """Per layer auxiliary loss.

        Args:
            aux_outputs (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
            insts (List):
                Ground truth of len batch_size, each InstanceData_ with
                    `sp_masks` of shape (n_gts_i, n_points_i)
                    `labels_3d` of shape (n_gts_i,)
                    `query_masks` of shape (n_gts_i, n_queries_i).
        
        Returns:
            Tensor: loss value.
        """
        cls_preds = aux_outputs['cls_preds']
        pred_scores = aux_outputs['scores']
        pred_masks = aux_outputs['masks']
        pred_centers = aux_outputs['centers']
        pred_sizes = aux_outputs['sizes']

        if indices is None:
            indices = []
            for i in range(len(insts)):
                if pred_centers[i] is None:
                    pred_instances = InstanceData(
                        scores=cls_preds[i],
                        masks=pred_masks[i])
                else:
                    pred_instances = InstanceData(
                        scores=cls_preds[i],
                        masks=pred_masks[i],
                        centers=pred_centers[i],
                        sizes=pred_sizes[i])
                gt_instances = InstanceData(
                    labels=insts[i].labels_3d,
                    masks=insts[i].sp_masks)
                if hasattr(insts[i], 'query_masks'):
                    gt_instances.query_masks = insts[i].query_masks
                if hasattr(insts[i], 'instance_centers'):
                    gt_instances.instance_centers = insts[i].instance_centers
                if hasattr(insts[i], 'instance_sizes'):
                    gt_instances.instance_sizes = insts[i].instance_sizes
                indices.append(self.matcher(pred_instances, gt_instances))

        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_losses.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)))
        cls_loss = torch.mean(torch.stack(cls_losses))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses, center_losses, size_losses = [], [], [], [], []
        for mask, score, center, size, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores, pred_centers, pred_sizes,
                                                      insts, indices):
            # if len(inst) == 0:
            #     continue

            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
            pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

            if center is not None:
                pred_center = center[idx_q]
                tgt_center = inst.instance_centers[idx_gt, :3]
                center_losses.append(F.l1_loss(pred_center, tgt_center, reduction='none').sum(-1).mean())
            if size is not None:
                pred_size = size[idx_q]
                tgt_size = inst.instance_sizes[idx_gt, :3]
                size_losses.append(F.l1_loss(pred_size, tgt_size, reduction='none').sum(-1).mean())
            
            # check if skip objectness loss
            if score is None:
                continue

            pred_score = score[idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss  = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0

        if len(center_losses):
            center_loss = torch.stack(center_losses).mean()
        else:
            center_loss = 0
        if len(size_losses):
            size_loss = torch.stack(size_losses).mean()
        else:
            size_loss = 0

        if len(self.loss_weight) == 4:
            loss = (
                self.loss_weight[0] * cls_loss +
                self.loss_weight[1] * mask_bce_loss +
                self.loss_weight[2] * mask_dice_loss +
                self.loss_weight[3] * score_loss)
        elif len(self.loss_weight) == 5:
            loss = (
                self.loss_weight[0] * cls_loss +
                self.loss_weight[1] * mask_bce_loss +
                self.loss_weight[2] * mask_dice_loss +
                self.loss_weight[3] * score_loss +
                self.loss_weight[4] * center_loss)
        elif len(self.loss_weight) == 6:
            loss = (
                self.loss_weight[0] * cls_loss +
                self.loss_weight[1] * mask_bce_loss +
                self.loss_weight[2] * mask_dice_loss +
                self.loss_weight[3] * score_loss +
                self.loss_weight[4] * center_loss +
                self.loss_weight[5] * size_loss)

        return loss

    def __call__(self, pred, insts):
        """Loss main function.

        Args:
            pred (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
                Dict `aux_preds` with list of cls_preds, scores, and masks.
            insts (List):
                Ground truth of len batch_size, each InstanceData_ with
                    `sp_masks` of shape (n_gts_i, n_points_i)
                    `labels_3d` of shape (n_gts_i,)
                    `query_masks` of shape (n_gts_i, n_queries_i).
        
        Returns:
            Dict: with instance loss value.
        """
        cls_preds = pred['cls_preds']
        pred_scores = pred['scores']
        pred_masks = pred['masks']
        pred_centers = pred['centers']
        pred_sizes = pred['sizes']

        # match
        indices = []
        for i in range(len(insts)):
            if pred_centers[i] is None:
                pred_instances = InstanceData(
                    scores=cls_preds[i],
                    masks=pred_masks[i])
            else:
                pred_instances = InstanceData(
                    scores=cls_preds[i],
                    masks=pred_masks[i],
                    centers=pred_centers[i],
                    sizes=pred_sizes[i])
            gt_instances = InstanceData(
                labels=insts[i].labels_3d,
                masks=insts[i].sp_masks)
            if hasattr(insts[i], 'query_masks'):
                gt_instances.query_masks = insts[i].query_masks
            if hasattr(insts[i], 'instance_centers'):
                gt_instances.instance_centers = insts[i].instance_centers
            if hasattr(insts[i], 'instance_sizes'):
                gt_instances.instance_sizes = insts[i].instance_sizes
            indices.append(self.matcher(pred_instances, gt_instances))

        # class loss
        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_losses.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)))
        cls_loss = torch.mean(torch.stack(cls_losses))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses, center_losses, size_losses = [], [], [], [], []
        for mask, score, center, size, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores, pred_centers, pred_sizes,
                                                      insts, indices):
            # if len(inst) == 0:
            #     continue
            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
                pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

            if center is not None:
                pred_center = center[idx_q]
                tgt_center = inst.instance_centers[idx_gt, :3]
                center_losses.append(F.l1_loss(pred_center, tgt_center, reduction='none').sum(-1).mean())
            if size is not None:
                pred_size = size[idx_q]
                tgt_size = inst.instance_sizes[idx_gt, :3]
                size_losses.append(F.l1_loss(pred_size, tgt_size, reduction='none').sum(-1).mean())

            # check if skip objectness loss
            if score is None:
                continue

            pred_score = score[idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0
        
        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum()

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss  = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0

        if len(center_losses):
            center_loss = torch.stack(center_losses).mean()
        else:
            center_loss = 0
        if len(size_losses):
            size_loss = torch.stack(size_losses).mean()
        else:
            size_loss = 0

        if len(self.loss_weight) == 4:
            loss = (
                self.loss_weight[0] * cls_loss +
                self.loss_weight[1] * mask_bce_loss +
                self.loss_weight[2] * mask_dice_loss +
                self.loss_weight[3] * score_loss)
        elif len(self.loss_weight) == 5:
            loss = (
                self.loss_weight[0] * cls_loss +
                self.loss_weight[1] * mask_bce_loss +
                self.loss_weight[2] * mask_dice_loss +
                self.loss_weight[3] * score_loss +
                self.loss_weight[4] * center_loss)
        elif len(self.loss_weight) == 6:
            loss = (
                self.loss_weight[0] * cls_loss +
                self.loss_weight[1] * mask_bce_loss +
                self.loss_weight[2] * mask_dice_loss +
                self.loss_weight[3] * score_loss +
                self.loss_weight[4] * center_loss +
                self.loss_weight[5] * size_loss)
        if 'aux_outputs' in pred:
            if self.iter_matcher:
                indices = None
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss += self.get_layer_loss(aux_outputs, insts, indices)

        return {'inst_loss': loss}


@LOSSES.register_module()
class ScanNetUnifiedCriterion:
    """Simply call semantic and instance criterions.

    Args:
        num_semantic_classes (int): Number of semantic classes.
        sem_criterion (ConfigDict): Class for semantic loss calculation.
        inst_criterion (ConfigDict): Class for instance loss calculation.
    """

    def __init__(self, num_semantic_classes, sem_criterion, inst_criterion):
        self.num_semantic_classes = num_semantic_classes
        assert sem_criterion.pop('type', None) == 'ScanNetSemanticCriterion', \
            "Semantic criterion only support 'ScanNetSemanticCriterion' type currently."
        assert inst_criterion.pop('type', None) == 'InstanceCriterion', \
            "Instance criterion only support 'InstanceCriterion' type currently."
        self.sem_criterion = ScanNetSemanticCriterion(**sem_criterion)
        self.inst_criterion = InstanceCriterion(**inst_criterion)
    
    def __call__(self, pred, insts):
        """Calculate loss.

        Args:
            pred (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
                Dict `aux_preds` with list of cls_preds, scores, and masks
                List `sem_preds` of len batch_size each of shape
                    (n_queries, n_classes + 1).
            insts (list): Ground truth of len batch_size,
                each InstanceData_ with
                    `sp_masks` of shape (n_gts_i + n_classes + 1, n_points_i)
                    `labels_3d` of shape (n_gts_i + n_classes + 1,)
                    `query_masks` of shape
                        (n_gts_i + n_classes + 1, n_queries_i).

        Returns:
            Dict: with semantic and instance loss values.
        """
        sem_gts = []
        inst_gts = []
        n = self.num_semantic_classes

        for i in range(len(pred['masks'])):
            sem_gt = InstanceData()
            sem_gt.sp_masks = insts[i]["query_inst_sem_masks"][-n - 1:, :]  # num_classes + 1, n_super_points
            sem_gts.append(sem_gt)
            
            inst_gt = InstanceData()
            inst_gt.sp_masks = insts[i]["sp_inst_sem_masks"][:-n - 1, :]
            inst_gt.labels_3d = insts[i]["labels"]
            if hasattr(insts[i], 'instance_centers'):
                inst_gt.instance_centers = insts[i]["instance_centers"]
            if hasattr(insts[i], 'instance_sizes'):
                inst_gt.instance_sizes = insts[i]["instance_sizes"]
            if hasattr(insts[i], 'query_inst_sem_masks'):
                inst_gt.query_masks = insts[i]["query_inst_sem_masks"][:-n - 1, :]
            inst_gts.append(inst_gt)
            
        loss = {}
        loss_semantic = self.sem_criterion(pred, sem_gts)
        loss.update(loss_semantic)
        loss_instance = self.inst_criterion(pred, inst_gts)
        loss.update(loss_instance)
        return loss
