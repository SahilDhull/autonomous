"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import numpy as np
from Sim_ATAV.vehicle_control.controller_commons.perf_evaluation.detection_evaluation_config \
    import DetectionEvaluationConfig
from Sim_ATAV.vehicle_control.controller_commons.sensor_detection_tools import match_objects_in_sets


class DetectionPerformanceMonitor(object):
    def __init__(self, ground_truth_detector):
        self.gt_detector = ground_truth_detector
        self.monitor_list = []
        self.perception_system = None
        self.evaluation_list = []

    def create_monitor(self, sensor_type='overall', sensor_id=0,
                       target_objs=None, eval_type='localization', eval_alg=None):
        self.monitor_list.append(DetectionEvaluationConfig(sensor_type=sensor_type,
                                                           sensor_id=sensor_id,
                                                           target_objs=target_objs,
                                                           eval_type=eval_type,
                                                           eval_alg=eval_alg))
        return len(self.monitor_list) - 1

    def add_monitor(self, monitor):
        if isinstance(monitor, DetectionEvaluationConfig):
            self.monitor_list.append(monitor)
        else:
            self.monitor_list.append(DetectionEvaluationConfig())
        return len(self.monitor_list) - 1

    def add_target_obj_to_monitor(self, obj_type, obj_id, monitor_idx=-1):
        self.monitor_list[monitor_idx].add_target_object(object_type=obj_type, object_id=obj_id)

    def set_perception_system(self, perception_system):
        self.perception_system = perception_system

    def evaluate_detections(self):
        self.evaluation_list = []
        for monitor in self.monitor_list:
            for target_obj in monitor.target_objs:
                # monitor.sensor_id is discarded for now. It is for future extensions to allow evaluation of
                # multiple sensors of same kind.
                if monitor.sensor_type == 'overall':
                    detections = self.perception_system.new_detections
                elif monitor.sensor_type == 'lidar':
                    detections = self.perception_system.object_detector.lidar_objects
                elif monitor.sensor_type == 'radar':
                    detections = self.perception_system.object_detector.radar_objects
                elif monitor.sensor_type == 'camera':
                    detections = self.perception_system.object_detector.camera_objects
                else:
                    detections = []
                (matches, unmatched_gt, unmatched_det) = match_objects_in_sets(self.gt_detector.detected_objects,
                                                                               detections,
                                                                               distance_threshold=10.0,
                                                                               discard_unmatched_set_b=False)
                match_dict = dict(matches)
                obj_gt_ind = self.gt_detector.detection_index_dict[target_obj]
                if obj_gt_ind in match_dict:
                    obj_det_ind = match_dict[obj_gt_ind]
                    eval_result = self.evaluate(gt_obj=self.gt_detector.detected_objects[obj_gt_ind],
                                                det_obj=detections[obj_det_ind],
                                                eval_type=monitor.eval_type,
                                                eval_alg=monitor.eval_alg)
                else:
                    # Not detected.
                    eval_result = -1.0
                self.evaluation_list.append(eval_result)
        return self.evaluation_list

    def evaluate(self, gt_obj, det_obj, eval_type, eval_alg):
        eval_result = -1.0
        if eval_alg == 'd_square':
            eval_result = np.linalg.norm(np.array(gt_obj.object_position) - np.array(det_obj.object_position))
        return eval_result

    def get_evaluations(self):
        return self.evaluation_list
