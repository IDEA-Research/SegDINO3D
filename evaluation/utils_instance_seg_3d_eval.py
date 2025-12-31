# Copied from mmdet3d/evaluation/functional/instance_seg_eval.py
# We fix instance seg metric to accept boolean instance seg mask of
# shape (n_points, n_instances) instead of integer mask of shape
# (n_points, ).
import numpy as np
from mmengine.logging import print_log
from terminaltables import AsciiTable

# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py # noqa
from copy import deepcopy

import numpy as np

from mmdet3d.evaluation.functional.scannet_utils import util_3d


def evaluate_matches(matches, class_labels, options):
    """Evaluate instance segmentation from matched gt and predicted instances
    for all scenes.

    Args:
        matches (dict): Contains gt2pred and pred2gt infos for every scene.
        class_labels (tuple[str]): Class names.
        options (dict): ScanNet evaluator options. See get_options.

    Returns:
        np.array: Average precision scores for all thresholds and categories.
    """
    overlaps = options['overlaps']
    min_region_sizes = [options['min_region_sizes'][0]]
    dist_threshes = [options['distance_threshes'][0]]
    dist_confs = [options['distance_confs'][0]]

    # results: class x overlap
    ap = np.zeros((len(dist_threshes), len(class_labels), len(overlaps)),
                  float)
    pr_rc = np.zeros((2, len(class_labels), len(overlaps)),
                  float)
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for label_name in class_labels:
                    for p in matches[m]['pred'][label_name]:
                        if 'filename' in p:
                            pred_visited[p['filename']] = False
            for li, label_name in enumerate(class_labels):
                y_true = np.empty(0)    # 1 for TP, 0 for FP
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]
                    # filter groups in ground truth
                    gt_instances = [
                        gt for gt in gt_instances
                        if gt['vert_count'] >=
                        min_region_size and gt['med_dist'] <= distance_thresh
                        and gt['dist_conf'] >= distance_conf
                    ]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float('inf'))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)
                    # collect matches
                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False
                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['filename']]:
                                continue
                            overlap = float(pred['intersection']) / (
                                gt['vert_count'] + pred['vert_count'] -
                                pred['intersection'])
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive # noqa
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['filename']] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match]
                    cur_score = cur_score[cur_match]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:
                            overlap = float(gt['intersection']) / (
                                gt['vert_count'] + pred['vert_count'] -
                                gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                # group?
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count'] < min_region_size or gt[
                                        'med_dist'] > distance_thresh or gt[
                                            'dist_conf'] < distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(
                                num_ignore) / pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred['confidence']
                                cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(
                        y_score_sorted, return_index=True)
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    # follow https://github.com/ScanNet/ScanNet/pull/26 ? # noqa
                    num_true_examples = y_true_sorted_cumsum[-1] if len(
                        y_true_sorted_cumsum) > 0 else 0
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall[-1] = 0.

                    #compute optimal precision and recall, based on f1_score
                    f1_score = 2 * precision * recall / (precision + recall + 0.0001)
                    f1_argmax = f1_score.argmax()
                    best_pr = precision[f1_argmax]
                    best_rc = recall[f1_argmax]

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0],
                                                recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)

                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5],
                                             'valid')
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                    best_pr = 0
                    best_rc = 0
                else:
                    ap_current = float('nan')
                    best_pr = float('nan')
                    best_rc = float('nan')
                ap[di, li, oi] = ap_current
                pr_rc[0, li, oi] = best_pr
                pr_rc[1, li, oi] = best_rc

    return ap, pr_rc


def compute_averages(aps, pr_rc, options, class_labels):
    """Averages AP scores for all categories.

    Args:
        aps (np.array): AP scores for all thresholds and categories.
        options (dict): ScanNet evaluator options. See get_options.
        class_labels (tuple[str]): Class names.

    Returns:
        dict: Overall and per-category AP scores.
    """
    d_inf = 0
    o50 = np.where(np.isclose(options['overlaps'], 0.5))
    o25 = np.where(np.isclose(options['overlaps'], 0.25))
    o_all_but25 = np.where(
        np.logical_not(np.isclose(options['overlaps'], 0.25)))
    avg_dict = {}
    avg_dict['all_ap'] = np.nanmean(aps[d_inf, :, o_all_but25])
    avg_dict['all_ap_50%'] = np.nanmean(aps[d_inf, :, o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[d_inf, :, o25])
    avg_dict['all_prec_50%'] = np.nanmean(pr_rc[0, :, o50])
    avg_dict['all_rec_50%'] = np.nanmean(pr_rc[1, :, o50])
    avg_dict['classes'] = {}
    for (li, label_name) in enumerate(class_labels):
        avg_dict['classes'][label_name] = {}
        avg_dict['classes'][label_name]['ap'] = np.average(aps[d_inf, li,
                                                               o_all_but25])
        avg_dict['classes'][label_name]['ap50%'] = np.average(aps[d_inf, li,
                                                                  o50])
        avg_dict['classes'][label_name]['ap25%'] = np.average(aps[d_inf, li,
                                                                  o25])
        avg_dict['classes'][label_name]['prec50%'] = np.average(pr_rc[0, li,
                                                                  o50])
        avg_dict['classes'][label_name]['rec50%'] = np.average(pr_rc[1, li,
                                                                  o50])
    
    head_cats_scannet_200 = ['tv stand', 'curtain', 'blinds', 'shower curtain', 'bookshelf', 'tv', 'kitchen cabinet', 
                             'pillow', 'lamp', 'dresser', 'monitor', 'object', 'ceiling', 'board', 'stove', 'closet wall', 
                             'couch', 'office chair', 'kitchen counter', 'shower', 'closet', 'doorframe', 'sofa chair', 
                             'mailbox', 'nightstand', 'washing machine', 'picture', 'book', 'sink', 'recycling bin', 
                             'table', 'backpack', 'shower wall', 'toilet', 'copier', 'counter', 'stool', 'refrigerator', 
                             'window', 'file cabinet', 'chair', 'plant', 'coffee table', 'stairs', 'armchair', 'cabinet', 
                             'bathroom vanity', 'bathroom stall', 'mirror', 'blackboard', 'trash can', 'stair rail', 
                             'box', 'towel', 'door', 'clothes', 'whiteboard', 'bed', 'bathtub', 'desk', 'wardrobe', 
                             'clothes dryer', 'radiator', 'shelf']
    
    common_cats_scannet_200 = ["cushion", "end table", "dining table", "keyboard", "bag", "toilet paper", "printer", 
                            "blanket", "microwave", "shoe", "computer tower", "bottle", "bin", "ottoman", "bench", 
                            "basket", "fan", "laptop", "person", "paper towel dispenser", "oven", "rack", "piano", 
                            "suitcase", "rail", "container", "telephone", "stand", "light", "laundry basket", 
                            "pipe", "seat", "column", "bicycle", "ladder", "jacket", "storage bin", "coffee maker", 
                            "dishwasher", "machine", "mat", "windowsill", "bulletin board", "fireplace", "mini fridge", 
                            "water cooler", "shower door", "pillar", "ledge", "furniture", "cart", "decoration", 
                            "closet door", "vacuum cleaner", "dish rack", "range hood", "projector screen", "divider", 
                            "bathroom counter", "laundry hamper", "bathroom stall door", "ceiling light", "trash bin", 
                            "bathroom cabinet", "structure", "storage organizer", "potted plant", "mattress"]
                            
    tail_cats_scannet_200 = ["paper", "plate", "soap dispenser", "bucket", "clock", "guitar", "toilet paper holder", 
                            "speaker", "cup", "paper towel roll", "bar", "toaster", "ironing board", "soap dish", 
                            "toilet paper dispenser", "fire extinguisher", "ball", "hat", "shower curtain rod", 
                            "paper cutter", "tray", "toaster oven", "mouse", "toilet seat cover dispenser", 
                            "storage container", "scale", "tissue box", "light switch", "crate", "power outlet", 
                            "sign", "projector", "candle", "plunger", "stuffed animal", "headphones", "broom", 
                            "guitar case", "dustpan", "hair dryer", "water bottle", "handicap bar", "purse", "vent", 
                            "shower floor", "water pitcher", "bowl", "paper bag", "alarm clock", "music stand", 
                            "laundry detergent", "dumbbell", "tube", "cd case", "closet rod", "coffee kettle", 
                            "shower head", "keyboard piano", "case of water bottles", "coat rack", "folded chair", 
                            "fire alarm", "power strip", "calendar", "poster", "luggage"]
    
    head_indices = [i for i, cls in enumerate(class_labels) if cls in head_cats_scannet_200]
    avg_dict['head_ap'] = np.nanmean(aps[d_inf][np.ix_(head_indices, o_all_but25[0])])
    avg_dict['head_ap_50%'] = np.nanmean(aps[d_inf][np.ix_(head_indices, o50[0])])
    avg_dict['head_ap_25%'] = np.nanmean(aps[d_inf][np.ix_(head_indices, o25[0])])
    avg_dict['head_prec_50%'] = np.nanmean(pr_rc[0][np.ix_(head_indices, o50[0])])
    avg_dict['head_rec_50%'] = np.nanmean(pr_rc[1][np.ix_(head_indices, o50[0])])

    common_indices = [i for i, cls in enumerate(class_labels) if cls in common_cats_scannet_200]
    avg_dict['common_ap'] = np.nanmean(aps[d_inf][np.ix_(common_indices, o_all_but25[0])])
    avg_dict['common_ap_50%'] = np.nanmean(aps[d_inf][np.ix_(common_indices, o50[0])])
    avg_dict['common_ap_25%'] = np.nanmean(aps[d_inf][np.ix_(common_indices, o25[0])])
    avg_dict['common_prec_50%'] = np.nanmean(pr_rc[0][np.ix_(common_indices, o50[0])])
    avg_dict['common_rec_50%'] = np.nanmean(pr_rc[1][np.ix_(common_indices, o50[0])])
    
    tail_indices = [i for i, cls in enumerate(class_labels) if cls in tail_cats_scannet_200]
    avg_dict['tail_ap'] = np.nanmean(aps[d_inf][np.ix_(tail_indices, o_all_but25[0])])
    avg_dict['tail_ap_50%'] = np.nanmean(aps[d_inf][np.ix_(tail_indices, o50[0])])
    avg_dict['tail_ap_25%'] = np.nanmean(aps[d_inf][np.ix_(tail_indices, o25[0])])
    avg_dict['tail_prec_50%'] = np.nanmean(pr_rc[0][np.ix_(tail_indices, o50[0])])
    avg_dict['tail_rec_50%'] = np.nanmean(pr_rc[1][np.ix_(tail_indices, o50[0])])

    return avg_dict


def assign_instances_for_scan(pred_info, gt_ids, options, valid_class_ids,
                              class_labels, id_to_label):
    """Assign gt and predicted instances for a single scene.

    Args:
        pred_info (dict): Predicted masks, labels and scores.
        gt_ids (np.array): Ground truth instance masks.
        options (dict): ScanNet evaluator options. See get_options.
        valid_class_ids (tuple[int]): Ids of valid categories.
        class_labels (tuple[str]): Class names.
        id_to_label (dict[int, str]): Mapping of valid class id to class label.

    Returns:
        dict: Per class assigned gt to predicted instances.
        dict: Per class assigned predicted to gt instances.
    """
    # get gt instances
    gt_instances = util_3d.get_instances(gt_ids, valid_class_ids, class_labels,
                                         id_to_label)
    # associate
    gt2pred = deepcopy(gt_instances)
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt['matched_pred'] = []
    pred2gt = {}
    for label in class_labels:
        pred2gt[label] = []
    num_pred_instances = 0
    # mask of void labels in the ground truth
    bool_void = np.logical_not(np.in1d(gt_ids // 1000, valid_class_ids))
    # go through all prediction masks
    for pred_mask_file in pred_info:
        label_id = int(pred_info[pred_mask_file]['label_id'])
        conf = pred_info[pred_mask_file]['conf']
        if not label_id in id_to_label:  # noqa E713
            continue
        label_name = id_to_label[label_id]
        # read the mask
        pred_mask = pred_info[pred_mask_file]['mask']
        if len(pred_mask) != len(gt_ids):
            raise ValueError('len(pred_mask) != len(gt_ids)')
        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < options['min_region_sizes'][0]:
            continue  # skip if empty

        pred_instance = {}
        pred_instance['filename'] = pred_mask_file
        pred_instance['pred_id'] = num_pred_instances
        pred_instance['label_id'] = label_id
        pred_instance['vert_count'] = num
        pred_instance['confidence'] = conf
        pred_instance['void_intersection'] = np.count_nonzero(
            np.logical_and(bool_void, pred_mask))   # the intersection with void points(points' gt class not in valid_class_ids)

        # matched gt instances
        matched_gt = []
        # go through all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(
                np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy['intersection'] = intersection
                pred_copy['intersection'] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
        pred_instance['matched_gt'] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt


def scannet_eval(preds, gts, options, valid_class_ids, class_labels,
                 id_to_label):
    """Evaluate instance segmentation in ScanNet protocol.

    Args:
        preds (list[dict]): Per scene predictions of mask, label and
            confidence.
        gts (list[np.array]): Per scene ground truth instance masks.
        options (dict): ScanNet evaluator options. See get_options.
        valid_class_ids (tuple[int]): Ids of valid categories.
        class_labels (tuple[str]): Class names.
        id_to_label (dict[int, str]): Mapping of valid class id to class label.

    Returns:
        dict: Overall and per-category AP scores.
    """
    options = get_options(options)
    matches = {}
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        matches_key = i
        # assign gt to predictions
        gt2pred, pred2gt = assign_instances_for_scan(pred, gt, options,
                                                     valid_class_ids,
                                                     class_labels, id_to_label)
        matches[matches_key] = {}
        matches[matches_key]['gt'] = gt2pred
        matches[matches_key]['pred'] = pred2gt

    ap_scores, pr_rc = evaluate_matches(matches, class_labels, options)
    avgs = compute_averages(ap_scores, pr_rc, options, class_labels)
    return avgs


def get_options(options=None):
    """Set ScanNet evaluator options.

    Args:
        options (dict, optional): Not default options. Default: None.

    Returns:
        dict: Updated options with all 4 keys.
    """
    assert options is None or isinstance(options, dict)
    _options = dict(
        overlaps=np.append(np.arange(0.5, 0.95, 0.05), 0.25),
        min_region_sizes=np.array([100]),
        distance_threshes=np.array([float('inf')]),
        distance_confs=np.array([-float('inf')]))
    if options is not None:
        _options.update(options)
    return _options



# 1) We fix this line: info[file_name]['mask'] = mask[i].
# 2) mask.max() + 1 in for is always equal to 2.
#    We have changed it to mask.shape[0] for iterating over all masks.
def aggregate_predictions(masks, labels, scores, valid_class_ids):
    """Maps predictions to ScanNet evaluator format.

    Args:
        masks (list[torch.Tensor]): Per scene predicted instance masks.
        labels (list[torch.Tensor]): Per scene predicted instance labels.
        scores (list[torch.Tensor]): Per scene predicted instance scores.
        valid_class_ids (tuple[int]): Ids of valid categories.

    Returns:
        list[dict]: Per scene aggregated predictions.
    """
    infos = []
    for id, (mask, label, score) in enumerate(zip(masks, labels, scores)):
        mask = mask.numpy()
        label = label.numpy()
        score = score.numpy()
        info = dict()
        for i in range(mask.shape[0]):
            # match pred_instance['filename'] from assign_instances_for_scan
            file_name = f'{id}_{i}'
            info[file_name] = dict()
            info[file_name]['mask'] = mask[i]   # (num_points, )
            info[file_name]['label_id'] = valid_class_ids[label[i]]
            info[file_name]['conf'] = score[i]
        infos.append(info)
    return infos


# For some reason the inputs are not torch.Tensor but np.ndarray.
# We just remove torch -> numpy conversion here.
def rename_gt(gt_semantic_masks, gt_instance_masks, valid_class_ids):
    """Maps gt instance and semantic masks to instance masks for ScanNet
    evaluator.

    Args:
        gt_semantic_masks (list[np.ndarray]): Per scene gt semantic masks.
        gt_instance_masks (list[np.ndarray]): Per scene gt instance masks.
        valid_class_ids (tuple[int]): Ids of valid categories.

    Returns:
        list[np.array]: Per scene instance masks.
    """
    renamed_instance_masks = []
    for semantic_mask, instance_mask in zip(gt_semantic_masks,
                                            gt_instance_masks):
        unique = np.unique(instance_mask)
        assert len(unique) < 1000
        for i in unique:
            semantic_instance = semantic_mask[instance_mask == i]
            semantic_unique = np.unique(semantic_instance)
            assert len(semantic_unique) == 1
            if semantic_unique[0] in valid_class_ids:
                instance_mask[instance_mask ==
                              i] = 1000 * semantic_unique[0] + i
        renamed_instance_masks.append(instance_mask)
    return renamed_instance_masks


def instance_seg_eval(gt_semantic_masks,
                      gt_instance_masks,
                      pred_instance_masks,
                      pred_instance_labels,
                      pred_instance_scores,
                      valid_class_ids,
                      class_labels,
                      options=None,
                      logger=None,
                      print_log_flag=True):
    """Instance Segmentation Evaluation.

    Evaluate the result of the instance segmentation.

    Args:
        gt_semantic_masks (list[torch.Tensor]): Ground truth semantic masks.
        gt_instance_masks (list[torch.Tensor]): Ground truth instance masks.
        pred_instance_masks (list[torch.Tensor]): Predicted instance masks.
        pred_instance_labels (list[torch.Tensor]): Predicted instance labels.
        pred_instance_scores (list[torch.Tensor]): Predicted instance labels.
        valid_class_ids (tuple[int]): Ids of valid categories.
        class_labels (tuple[str]): Names of valid categories.
        options (dict, optional): Additional options. Keys may contain:
            `overlaps`, `min_region_sizes`, `distance_threshes`,
            `distance_confs`. Default: None.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Returns:
        dict[str, float]: Dict of results.
    """
    assert len(valid_class_ids) == len(class_labels)
    id_to_label = {
        valid_class_ids[i]: class_labels[i]
        for i in range(len(valid_class_ids))
    }
    preds = aggregate_predictions(
        masks=pred_instance_masks,
        labels=pred_instance_labels,
        scores=pred_instance_scores,
        valid_class_ids=valid_class_ids)
    gts = rename_gt(gt_semantic_masks, gt_instance_masks, valid_class_ids)
    metrics = scannet_eval(
        preds=preds,
        gts=gts,
        options=options,
        valid_class_ids=valid_class_ids,
        class_labels=class_labels,
        id_to_label=id_to_label)
    header = ['classes', 'AP_0.25', 'AP_0.50', 'AP', 'Prec_0.50', 'Rec_0.50']
    rows = []
    for label, data in metrics['classes'].items():
        aps = [data['ap25%'], data['ap50%'], data['ap'], data['prec50%'], data['rec50%']]
        rows.append([label] + [f'{ap:.4f}' for ap in aps])
    aps = metrics['all_ap_25%'], metrics['all_ap_50%'], metrics['all_ap'], metrics['all_prec_50%'], metrics['all_rec_50%']
    footer = ['Overall'] + [f'{ap:.4f}' for ap in aps]

    aps = metrics['head_ap_25%'], metrics['head_ap_50%'], metrics['head_ap'], metrics['head_prec_50%'], metrics['head_rec_50%']
    head_footer = ['Head Overall'] + [f'{ap:.4f}' for ap in aps]
    aps = metrics['common_ap_25%'], metrics['common_ap_50%'], metrics['common_ap'], metrics['common_prec_50%'], metrics['common_rec_50%']
    common_footer = ['Common Overall'] + [f'{ap:.4f}' for ap in aps]
    aps = metrics['tail_ap_25%'], metrics['tail_ap_50%'], metrics['tail_ap'], metrics['tail_prec_50%'], metrics['tail_rec_50%']
    tail_footer = ['Tail Overall'] + [f'{ap:.4f}' for ap in aps]

    table = AsciiTable([header] + rows + [head_footer] + [common_footer] + [tail_footer] + [footer])
    table.inner_footing_row_border = True
    if print_log_flag:
        print_log('\n' + table.table, logger=logger)
    return metrics
