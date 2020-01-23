"""
Provides functions for computing object detection and classification algorithm performance.
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


def iou_performance_for_object(ground_truth_bbox, detection_boxes,
                               detection_probs=None, prob_threshold=0.0,
                               obj_class=None, detection_classes=None):
    """Computes Intersection over Union (IoU) performance for detection of an object.
    Returns largest IoU performance among the detection boxes with the same class.
    If detection probabilities are also provided, IoU is multiplied with detection prob.
    Otherwise, max_perf = iou_value.
    If obj_class or detection_classes are not provided, class label is ignored.
    IoU is also known as Jaccard Index."""
    max_iou_ind = -1
    max_iou = 0
    max_perf = 0
    prob_full_range = 1.0 - prob_threshold
    for (box_ind, det_box) in enumerate(detection_boxes):
        # Check if class labels will be included in the computation
        if obj_class is None or detection_classes is None:
            gt_class = 0
            det_class = 0
        else:
            gt_class = obj_class
            det_class = detection_classes[box_ind]
        if gt_class == det_class:
            iou_value = intersection_over_union(ground_truth_bbox, det_box)
            # Check if probabilities will be included in the computation
            if detection_probs is not None:
                prob_perf = (detection_probs[box_ind] - prob_threshold) / prob_full_range
                prob_perf = min(max(prob_perf, 0.0), 1.0)  # Make sure it is between 0 and 1
            else:
                prob_perf = 1.0
        else:
            iou_value = 0.0
            prob_perf = 0.0

        perf_value = iou_value * prob_perf
        if perf_value > max_perf:
            max_perf = perf_value
            max_iou = iou_value
            max_iou_ind = box_ind
    return max_perf, max_iou, max_iou_ind


def intersection_over_union(ground_truth_bbox, detection_box):
    """Computes IoU between two boxes.
    We Scale it with 100. Otherwise, small numbers are lost when transfered from python to matlab.
    Boxes should be lists in the format [center x, center y, width, height]"""
    # right-most of the left edges:
    intersection_box_x_left = max(ground_truth_bbox[0] - ground_truth_bbox[2]/2, \
                                  detection_box[0] - detection_box[2]/2)
    # left-most of the right edges:
    intersection_box_x_right = min(ground_truth_bbox[0] + ground_truth_bbox[2]/2, \
                                   detection_box[0] + detection_box[2]/2)
    # up-most of the bottom edges:
    intersection_box_y_down = max(ground_truth_bbox[1] - ground_truth_bbox[3]/2, \
                                  detection_box[1] - detection_box[3]/2)
    # down-most of the top edges:
    intersection_box_y_up = min(ground_truth_bbox[1] + ground_truth_bbox[3]/2, \
                                detection_box[1] + detection_box[3]/2)
    if (intersection_box_y_up > intersection_box_y_down and
            intersection_box_x_right > intersection_box_x_left):
        intersection_area = (intersection_box_y_up - intersection_box_y_down) * \
                            (intersection_box_x_right - intersection_box_x_left)
        gt_area = ground_truth_bbox[2] * ground_truth_bbox[3]
        det_box_area = detection_box[2] * detection_box[3]
        if gt_area + det_box_area - intersection_area > 0.1:
            iou = 100.0 * intersection_area / (gt_area + det_box_area - intersection_area)
        else:
            iou = 100.0
        iou = min(max(iou, 0.0), 100.0)  # Make sure it is between 0 and 1
    else:
        iou = 0.0
    return iou
