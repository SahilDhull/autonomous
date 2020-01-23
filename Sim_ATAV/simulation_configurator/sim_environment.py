""" Defines SimEnvironment class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


class SimEnvironment(object):
    def __init__(self):
        self.fog = None
        self.heart_beat_config = None
        self.view_follow_config = None
        self.ego_vehicles_list = []
        self.agent_vehicles_list = []
        self.pedestrians_list = []
        self.road_list = []
        self.road_disturbances_list = []
        self.controller_params_list = []
        self.data_log_description_list = []
        self.data_log_period_ms = None
        self.detection_evaluation_config_list = []
        self.visibility_evaluation_config_list = []
        self.initial_state_config_list = []
        self.stop_before_collision_config_list = []
        self.periodic_reporting_config_list = []
        self.simulation_trace_dict = {}
        self.ego_target_path = []
        self.generic_sim_objects_list = []

    def clear_all(self):
        self.fog = None
        self.heart_beat_config = None
        self.view_follow_config = None
        self.ego_vehicles_list = []
        self.agent_vehicles_list = []
        self.pedestrians_list = []
        self.road_list = []
        self.road_disturbances_list = []
        self.controller_params_list = []
        self.data_log_description_list = []
        self.data_log_period_ms = None
        self.detection_evaluation_config_list = []
        self.visibility_evaluation_config_list = []
        self.initial_state_config_list = []
        self.stop_before_collision_config_list = []
        self.periodic_reporting_config_list = []
        self.simulation_trace_dict = {}
        self.ego_target_path = []
        self.generic_sim_objects_list = []

    def populate_simulation_trace_dict(self):
        if self.data_log_period_ms is not None:
            self.simulation_trace_dict['time_step'] = self.data_log_period_ms
        traj_ind = 0
        for (traj_ind, data_log_item) in enumerate(self.data_log_description_list):
            self.simulation_trace_dict[(data_log_item.item_type,
                                        data_log_item.item_index,
                                        data_log_item.item_state_index)] = traj_ind
        cur_traj_ind = traj_ind
        for detection_evaluation_config in self.detection_evaluation_config_list:
            for target_obj_ind in range(len(detection_evaluation_config.target_objs)):
                cur_traj_ind += 1
                self.simulation_trace_dict[
                    detection_evaluation_config.get_target_obj_info_as_dictionary_key(target_obj_ind)] = cur_traj_ind

        for visibility_config in self.visibility_evaluation_config_list:
            for target_obj_ind in range(len(visibility_config.object_list)):
                cur_traj_ind += 1
                self.simulation_trace_dict[
                    visibility_config.get_target_obj_info_as_dictionary_key(target_obj_ind)] = cur_traj_ind
