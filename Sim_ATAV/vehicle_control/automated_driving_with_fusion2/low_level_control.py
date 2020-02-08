"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import math


class LowLevelControl(object):
    def __init__(self, ego_state, longitudinal_controller, lateral_controller, path_planner):
        self.ego_state = ego_state
        self.longitudinal_controller = longitudinal_controller
        self.lateral_controller = lateral_controller
        self.long_position_offset = 0.0
        # path_planner can be a path planner object or just a tuple of target bearing and target lateral position
        if isinstance(path_planner, tuple):
            self.target_bearing = path_planner[0]
            self.target_lat_pos = path_planner[1]
            self.path_planner = None
        else:
            self.path_planner = path_planner
            self.target_lat_pos = 0.0
            self.target_bearing = 0.0

    def set_parameter(self, parameter_name, parameter_value):
        if parameter_name == "long_position_offset":
            # long_position_offset is used to add an offset to the longitudinal position for control computations.
            # This is useful for computing smoother turns.
            self.long_position_offset = parameter_value

    def mode_change_reset(self):
        self.longitudinal_controller.i_state = 0.0

    def compute_throttle_and_steering(self, current_target_speed, control_mode):
        """Computes control output using the detected objects from sensor suite."""
        # Compute control
        cur_position = self.ego_state.get_position()
        cur_yaw_angle = self.ego_state.get_yaw_angle()
        cur_speed_ms = self.ego_state.get_speed_ms()
        if self.path_planner.path_following_tools.target_path is not None:
            # Compute distance from front wheels for smoother turns:
            temp_cur_pos = [cur_position[0] - (self.long_position_offset * math.sin(cur_yaw_angle) +
                                               cur_speed_ms  * 0.2 * math.sin(cur_yaw_angle)),
                            cur_position[1] + (self.long_position_offset * math.cos(cur_yaw_angle) +
                                               cur_speed_ms * 0.2 * math.cos(cur_yaw_angle))]
            (distance_err, angle_err) = self.path_planner.get_distance_and_angle_error(cur_position=temp_cur_pos,
                                                                                       cur_yaw_angle=cur_yaw_angle)
        else:
            angle_err = self.target_bearing - cur_yaw_angle
            while angle_err > math.pi:
                angle_err -= 2 * math.pi
            while angle_err < -math.pi:
                angle_err += 2 * math.pi
            distance_err = -(self.target_lat_pos - cur_position[0])

        if control_mode == 'emergency':
            control_throttle = -1.0
        else:
            control_throttle = self.longitudinal_controller.compute(current_target_speed - cur_speed_ms)
            if control_mode == 'cautious':
                control_throttle = max(-0.7, control_throttle)
            elif control_mode == 'moderate':
                control_throttle = max(-0.5, control_throttle)
            else:
                control_throttle = max(-0.3, control_throttle)

        control_steering = self.lateral_controller.compute(angle_err, distance_err, cur_speed_ms)
        speed_ratio = min(1.0, cur_speed_ms / 22.0)
        max_steering = 0.1 + (1.0 - speed_ratio) * 0.7
        control_steering = min(max(-max_steering, control_steering), max_steering)
        # self.console_output.debug_print('steering: {} throttle: {}'.format(control_steering, control_throttle))
        return control_throttle, control_steering
