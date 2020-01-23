"""Defines sensor object detection related methods
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import numpy as np


def match_objects_in_sets(set_a, set_b, distance_threshold=5.0, discard_unmatched_set_b=True):
    """Function to find matching objects from two detection lists based on Euclidean distance between objects."""
    # This is a linear matching problem and there are better algorithms to solve this,
    # like Jonker-Volgenant algorithm or Hungarian algorithm.
    # This implementation is not even complete. It may miss matching some objects if there is
    # a conflict between closest objects from set_a to set_b and from set_b to set_a.
    # I leave this as an improvement since for the small number of objects and in most cases we should be fine.

    # This implementation declares a match only if:
    # (obj_b is the closest to obj_a in set_b) and (obj_a is closest to obj_b in set_a)
    matches = []
    unmatched_a = []
    unmatched_b = []
    if len(set_a) > 0 and len(set_b) == 0:
        unmatched_a = list(range(len(set_a)))
    elif len(set_a) == 0 and len(set_b) > 0:
        unmatched_b = list(range(len(set_b)))
    elif len(set_a) > 0 and len(set_b) > 0:
        dist_matrix = -1*np.ones([len(set_a), len(set_b)])
        for (obj_a_ind, obj_a) in enumerate(set_a):
            for (obj_b_ind, obj_b) in enumerate(set_b):
                if obj_a.sensor_recorded_position is not None and obj_b.sensor_recorded_position is not None:
                    dist_matrix[obj_a_ind, obj_b_ind] = \
                        np.linalg.norm(np.array(obj_a.sensor_recorded_position)
                                       - np.array(obj_b.sensor_recorded_position))
                else:
                    dist_matrix[obj_a_ind, obj_b_ind] = \
                        np.linalg.norm(np.array(obj_a.object_position) - np.array(obj_b.object_position))
        closest_obj_for_set_a = dist_matrix.argmin(axis=1)
        closest_obj_for_set_b = dist_matrix.argmin(axis=0)

        for obj_a_ind in range(len(set_a)):
            closest_b_ind = closest_obj_for_set_a[obj_a_ind]
            if (closest_obj_for_set_b[closest_b_ind] == obj_a_ind and
                    dist_matrix[obj_a_ind, closest_b_ind] < distance_threshold):
                matches.append([obj_a_ind, closest_b_ind])
            else:
                unmatched_a.append(obj_a_ind)
        matches = np.array(matches)
        if not discard_unmatched_set_b:
            for obj_b_ind in range(len(set_b)):
                if len(matches) + len(unmatched_b) < len(set_b):
                    break
                if obj_b_ind not in matches[:, 1]:
                    unmatched_b.append(obj_b_ind)
    return np.array(matches), np.array(unmatched_a), np.array(unmatched_b)
