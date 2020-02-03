"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from Sim_ATAV.vehicle_control.controller_commons.path_following_tools import PathFollowingTools
from Sim_ATAV.vehicle_control.controller_commons.planning.trajectory_estimation import TrajectoryEstimation
from Sim_ATAV.vehicle_control.controller_commons.visualization.console_output import ConsoleOutput


class PathPlanner(object):
    def __init__(self, console_output=None):
        self.path_following_tools = PathFollowingTools()
        self.is_path_modified = False
        if console_output is None:
            self.console_output = ConsoleOutput(debug_mode=False)
        else:
            self.console_output = console_output
        self.trajectory_estimation = TrajectoryEstimation(console_output,
                                                          path_following_tools=self.path_following_tools)
        self.last_segment_ind = 0
        self.dist_to_end_of_segment = 0.0

    def add_waypoint(self, waypoint_data):
        self.path_following_tools.add_point_to_path(waypoint_data)
        self.is_path_modified = True

    def apply_path_changes(self, force_apply=False):
        if self.is_path_modified or force_apply:
            # print(self.path_following_tools.target_path)
            self.path_following_tools.smoothen_the_path()
            # print(self.path_following_tools.target_path)
            self.path_following_tools.populate_the_path_with_details()
            # print(self.path_following_tools.path_details)
            self.is_path_modified = False

    def update_estimations(self, cur_position, cur_speed_ms, cur_yaw_angle, detected_objects):
        (self.last_segment_ind, line_segment_as_list, nearest_pos_on_path, self.dist_to_end_of_segment) = \
            self.path_following_tools.get_current_segment(vhc_pos=cur_position, last_segment_ind=self.last_segment_ind)
        self.trajectory_estimation.estimate_ego_future(cur_position,
                                                       cur_speed_ms,
                                                       cur_yaw_angle,
                                                       self.last_segment_ind)
        self.trajectory_estimation.estimate_future_ego_agent_conflicts(detected_objects=detected_objects,
                                                                       cur_ego_position=cur_position,
                                                                       cur_ego_yaw_angle=cur_yaw_angle)

    def get_distance_and_angle_error(self, cur_position, cur_yaw_angle):
        (distance_err, angle_err) = self.path_following_tools.get_distance_and_angle_error(
            vhc_pos=cur_position, vhc_bearing=cur_yaw_angle, last_segment_ind=self.last_segment_ind)
        return distance_err, angle_err

    def next_turn_information(self):
        if len(self.path_following_tools.path_details) > self.last_segment_ind:
            (next_turn_angle, travel_distance) = self.path_following_tools.path_details[self.last_segment_ind]
            travel_distance += self.dist_to_end_of_segment
        else:
            (next_turn_angle, travel_distance) = (0.0, 0.0)
        return next_turn_angle, travel_distance
