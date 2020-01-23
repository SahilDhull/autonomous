"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from Sim_ATAV.vehicle_control.controller_commons.perf_evaluation.visibility_evaluator \
    import VisibilityEvaluator, VisibilityConfig, VisibilityRatio


class VisibilityMonitor(object):
    def __init__(self, ground_truth_detector):
        self.gt_detector = ground_truth_detector
        self.visibility_dict = {}
        self.monitor_list = []
        self.visibility_evaluator = VisibilityEvaluator()

    def add_monitor(self, monitor):
        if isinstance(monitor, VisibilityConfig):
            self.monitor_list.append(monitor)
            self.visibility_evaluator.add_sensor(monitor.sensor)

    def update_visibility_dict(self):
        for (monitor_ind, monitor) in enumerate(self.monitor_list):
            obj_visibility_dict = self.visibility_evaluator.get_all_vhc_and_ped_visibility_info(
                self_vhc_id=self.gt_detector.ego_vhc_id,
                sensor_ind=monitor_ind,
                vhc_pos_dict=self.gt_detector.vhc_pos_dict,
                vhc_rot_dict=self.gt_detector.vhc_rot_dict,
                vhc_corners_dict=self.gt_detector.vhc_corners_dict,
                ped_pos_dict=self.gt_detector.ped_pos_dict,
                ped_rot_dict=self.gt_detector.ped_rot_dict,
                ped_corners_dict=self.gt_detector.ped_corners_dict)
            for (obj_ind, obj) in enumerate(monitor.object_list):
                obj_key = (obj[VisibilityConfig.OBJ_ID_IND], obj[VisibilityConfig.OBJ_TYPE_IND])
                if obj_key in obj_visibility_dict:
                    self.visibility_dict[obj_key] = obj_visibility_dict[obj_key]
                else:
                    self.visibility_dict[obj_key] = -1.0

    def get_obj_visibility(self, obj_id, obj_type):
        if (obj_id, obj_type) in self.visibility_dict:
            visibility = self.visibility_dict[(obj_id, obj_type)]
        else:
            visibility = VisibilityRatio(0.0, 0.0)
        return visibility

    def get_evaluations(self):
        eval_list = []
        for monitor in self.monitor_list:
            for (obj_ind, obj) in enumerate(monitor.object_list):
                obj_key = (obj[VisibilityConfig.OBJ_ID_IND], obj[VisibilityConfig.OBJ_TYPE_IND])
                if obj_key in self.visibility_dict:
                    visibility = self.visibility_dict[obj_key]
                    eval_list.append(visibility.percent)
                else:
                    eval_list.append(-1.0)
        return eval_list
