"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import math
import numpy as np
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation.sensor_object import SensorObject


class HighLevelControl(object):
    def __init__(self, ego_state, low_level_controller, path_planner, console_output):
        self.control_mode = 'comfort'
        self.very_risky_object_list = []
        self.risky_object_list = []
        self.proceed_w_caution_object_list = []
        self.ego_state = ego_state
        self.low_level_controller = low_level_controller
        self.console_output = console_output
        self.risky_obj_distance_threshold = 15.0
        self.slow_down_at_intersections = True
        self.path_planner = path_planner
        self.target_speed_m_s = 0.0

    def set_parameter(self, parameter_name, parameter_value):
        if parameter_name == "risky_obj_distance_threshold":
            self.risky_obj_distance_threshold = parameter_value
        elif parameter_name == "slow_down_at_intersections":
            self.slow_down_at_intersections = parameter_value
        elif parameter_name == 'target_speed_m_s':
            self.target_speed_m_s = parameter_value

    def risk_assessment(self, sensor_detected_objects):
        cur_speed_ms = self.ego_state.get_speed_ms()
        # Check collision risk
        self.very_risky_object_list = []
        self.risky_object_list = []
        self.proceed_w_caution_object_list = []

        self.console_output.debug_print('Evaluating risk for the objects')
        for (obj_ind, detected_object) in enumerate(sensor_detected_objects):
            if 'future_intersection' in detected_object.aux_data:
                self.console_output.debug_print(
                    '({}): Type: {} Position: {} Direction: {} speed: {} future min dist: \
                    {} future time: {} future_lat:{}'.format(
                        obj_ind,
                        detected_object.object_type,
                        detected_object.object_position,
                        detected_object.object_direction,
                        detected_object.object_speed_m_s,
                        detected_object.aux_data['future_intersection'][0],
                        detected_object.aux_data['future_intersection'][1],
                        detected_object.aux_data['future_intersection'][2]))
                future_intersection = detected_object.aux_data['future_intersection']
                if future_intersection[0] < 5.0 and abs(future_intersection[2]) < 1.1 and future_intersection[1] < 2.51:
                    self.very_risky_object_list.append(obj_ind)
                    self.console_output.debug_print('({}) -> very risky (A)'.format(obj_ind))
                elif (future_intersection[0] < 6.0 and abs(future_intersection[2]) < 1.0 and
                      abs(future_intersection[1]) < 3.01):
                    self.risky_object_list.append(obj_ind)
                    self.console_output.debug_print('({}) -> risky (A)'.format(obj_ind))
                elif (future_intersection[0] < 8.0 and abs(future_intersection[2]) < 8.0 and
                      (np.sign(detected_object.aux_data['future_intersection'][2]) !=
                       np.sign(detected_object.object_position[0]))):
                    self.risky_object_list.append(obj_ind)
                    # object is passing from one side to the other
                    self.console_output.debug_print('({}) -> risky (A2)'.format(obj_ind))
                elif (future_intersection[0] < 7.0 and abs(future_intersection[2]) < 1.0 and
                      abs(future_intersection[1]) < 3.51):
                    self.proceed_w_caution_object_list.append(obj_ind)
                    self.console_output.debug_print('({}) -> caution (A)'.format(obj_ind))
                elif (future_intersection[0] < 12.0 and abs(future_intersection[2]) < 12.0 and
                      (np.sign(detected_object.aux_data['future_intersection'][2]) !=
                       np.sign(detected_object.object_position[0]))):
                    self.proceed_w_caution_object_list.append(obj_ind)
                    # object is passing from one side to the other
                    self.console_output.debug_print('({}) -> caution (A2)'.format(obj_ind))

        for (obj_ind, detected_object) in enumerate(sensor_detected_objects):
            y_intersection_time = detected_object.object_position[1] / \
                                  max(1.0, (-detected_object.object_direction[1] * detected_object.object_speed_m_s))
            if 100 > y_intersection_time > 0:
                x_travel_dist = \
                    detected_object.object_direction[0] * detected_object.object_speed_m_s * y_intersection_time
                x_intersection = detected_object.object_position[0] + x_travel_dist

                if SensorObject.SENSOR_LIDAR in detected_object.sensor_aux_data_dict:
                    # Because, Lidar's closest point and object center point are different.
                    cluster = detected_object.sensor_aux_data_dict[SensorObject.SENSOR_LIDAR].lidar_cluster
                    min_x = min(cluster.cluster_points[:, 0], key=abs)
                    obj_min_x = min(detected_object.object_position[0], min_x)
                    if abs(obj_min_x) < abs(x_intersection):
                        x_intersection = obj_min_x

                if ((detected_object.object_type == SensorObject.OBJECT_CAR and abs(x_intersection) < 2.0)
                        or (detected_object.object_type == SensorObject.OBJECT_PEDESTRIAN and abs(x_intersection) < 1.1)
                        or (detected_object.object_type == SensorObject.OBJECT_BIKE and abs(x_intersection) < 1.5)):
                    ttc = np.linalg.norm(detected_object.object_position) / max(1.0, detected_object.object_speed_m_s)
                    self.console_output.debug_print('ttc: {}'.format(ttc))
                    if ((ttc < 2.5 and detected_object.object_position[1] < 40.0) or
                            detected_object.object_position[1] < self.risky_obj_distance_threshold):
                        self.very_risky_object_list.append(obj_ind)
                        self.console_output.debug_print('({}) -> very risky (B)'.format(obj_ind))
                        # elif ttc < 4.0 and detected_object.object_position[1] < 80.0:
                    #     risky_object_list.append(obj_ind)
                    elif detected_object.object_position[1] < 40.0:
                        self.proceed_w_caution_object_list.append(obj_ind)
                        self.console_output.debug_print('({}) -> caution (B)'.format(obj_ind))
                else:
                    ttc = 1000.0
                self.console_output.debug_print(
                    '({}): Type: {} Position: {} Direction: {} speed: {} y_intersection_time: {} x intersection: {}'.format(
                        obj_ind,
                        detected_object.object_type,
                        detected_object.object_position,
                        detected_object.object_direction,
                        detected_object.object_speed_m_s,
                        y_intersection_time,
                        x_intersection))
            if (abs(detected_object.object_position[0]) < 1.0 and
                    (detected_object.object_position[1] < self.risky_obj_distance_threshold or
                     detected_object.object_position[1] < cur_speed_ms)):
                # whatever the ttc is. If there is object in front of you, stop.
                self.very_risky_object_list.append(obj_ind)
                self.console_output.debug_print('({}) -> very risky (C)'.format(obj_ind))

    def decide_control_mode(self):
        if self.very_risky_object_list:
            if self.control_mode != 'emergency':
                self.low_level_controller.mode_change_reset()
            self.control_mode = 'emergency'
        elif self.risky_object_list:
            if self.control_mode != 'cautious':
                self.low_level_controller.mode_change_reset()
            self.control_mode = 'cautious'
        elif self.proceed_w_caution_object_list:
            if self.control_mode != 'moderate':
                self.low_level_controller.mode_change_reset()
            if self.control_mode == 'emergency':
                self.control_mode = 'cautious'
            else:
                self.control_mode = 'moderate'
        else:
            if self.control_mode != 'comfort':
                self.low_level_controller.mode_change_reset()
            if self.control_mode == 'emergency':
                self.control_mode = 'cautious'
            elif self.control_mode == 'cautious':
                self.control_mode = 'moderate'
            else:
                self.control_mode = 'comfort'

    def compute_target_speed(self):
        if self.control_mode == 'emergency':
            current_target_speed = 0.0
        elif self.control_mode == 'cautious':
            current_target_speed = min(self.target_speed_m_s, min(8.0, self.ego_state.get_speed_ms()))
        elif self.control_mode == 'moderate':
            # If stopped, you can start proceeding with 2m/s
            current_target_speed = min(self.target_speed_m_s, min(15.0, max(self.ego_state.get_speed_ms(), 2.0)))
        else:
            current_target_speed = self.target_speed_m_s

        if self.path_planner is not None:
            (next_turn_angle, travel_distance) = self.path_planner.next_turn_information()
            if self.slow_down_at_intersections and abs(next_turn_angle) > math.pi / 60 and travel_distance < 100.0:
                turn_ratio = min(1.0, abs(next_turn_angle) / (math.pi / 4.0))
                max_speed_limit = 10.0 + ((1.0 - turn_ratio) * 30.0)
                # decrease speed limit as we approach to the intersection.
                max_speed_limit += (current_target_speed - max_speed_limit) * \
                                   ((max(travel_distance, 10.0) - 10.0) / 80.0)
                current_target_speed = min(current_target_speed, max_speed_limit)

        return current_target_speed

    def compute_control(self, detected_objects=None):
        self.risk_assessment(detected_objects)
        self.decide_control_mode()
        current_target_speed = self.compute_target_speed()
        # print(self.control_mode)
        # print("current_target_speed")
        # print(current_target_speed)
        (control_throttle, control_steering) = \
            self.low_level_controller.compute_throttle_and_steering(current_target_speed=current_target_speed,
                                                                    control_mode=self.control_mode)
        return control_throttle, control_steering
