"""Defines RobustnessComputation class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import numpy as np
from Sim_ATAV.common.coordinate_system import CoordinateSystem


class RobustnessComputation(object):
    """Handles the robustness computation"""
    ROB_TYPE_TTC_COLL_SPEED = 0
    ROB_TYPE_DUMMY_TRAIN_COLL = 1
    ROB_TYPE_DUMMY_TRAIN_DRIVE = 2
    ROB_TYPE_DUMMY_TRAIN_FRONT_COLL = 3
    ROB_TYPE_STAY_IN_THE_MIDDLE = 4
    ROB_TYPE_FOLLOW = 5
    ROB_TYPE_TRAFFIC_WAVE = 6
    ROB_TYPE_MAX_ABS_JERK = 7
    ROB_TYPE_DSSM_FRONT_REAR = 8
    ROB_TYPE_DISTANCE_TO_PEDESTRIAN = 10
    ROB_TYPE_NONE = 100
    MAX_ROBUSTNESS = 100
    MIN_COLL_MAGNITUDE = 0.0
    MAX_COLL_MAGNITUDE = 30.0
    TOUCH_SENSOR_BUMPER = -100
    TOUCH_SENSOR_FORCE = -200
    TOUCH_SENSOR_FORCE_3D = 0
    TOUCH_SENSOR_UNKNOWN_TYPE = -1

    def __init__(self, robustness_type=None, supervisor_control=None, vehicles_manager=None, environment_manager=None, pedestrians_manager=None):
        self.debug_mode = 0
        if robustness_type is None:
            robustness_type = self.ROB_TYPE_NONE
        self.robustness_type = robustness_type
        self.supervisor_control = supervisor_control
        self.vehicles_manager = vehicles_manager
        self.pedestrians_manager = pedestrians_manager
        self.environment_manager = environment_manager
        self.minimum_robustness = self.MAX_ROBUSTNESS
        self.minimum_robustness_instant = 0
        self.collision_detected = False
        self.touch_sensor_type = self.TOUCH_SENSOR_UNKNOWN_TYPE
        self.extra_punishment = 0.0
        self.started_computing_robustness = False
        self.traffic_wave_threshold_speed = 8.33  # Corresponds to 30 km/h #5.56 #: 20 km/h
        self.min_dummy_speed = self.MAX_ROBUSTNESS

    def set_debug_mode(self, mode):
        """Sets the debug mode for this object."""
        self.debug_mode = mode

    def compute_ang_wrt_pos_lat_axis(self, vector):
        """Computes the vector angle, i.e., is the angle of the vector wrt positive lateral axis"""
        if -0.00001 < vector[CoordinateSystem.LAT_AXIS] < 0.00001:
            if vector[CoordinateSystem.LONG_AXIS] >= 0:
                vector_ang = math.pi / 2.0
            else:
                vector_ang = -math.pi / 2.0
        elif vector[CoordinateSystem.LAT_AXIS] > 0:
            vector_ang = math.atan(vector[CoordinateSystem.LONG_AXIS] / vector[CoordinateSystem.LAT_AXIS])
        elif vector[CoordinateSystem.LONG_AXIS] < 0:  # vector[CoordinateSystem.LAT_AXIS] < 0
            vector_ang = -math.pi + math.atan(vector[CoordinateSystem.LONG_AXIS] / vector[CoordinateSystem.LAT_AXIS])
        else:  # vector[CoordinateSystem.LONG_AXIS] >= 0 and vector[CoordinateSystem.LAT_AXIS] < 0
            vector_ang = math.pi + math.atan(vector[CoordinateSystem.LONG_AXIS] / vector[CoordinateSystem.LAT_AXIS])
        return vector_ang

    def check_collision_course(self, ego_vhc_pos, ego_vhc_pts, ego_vhc_ang_velocity, ego_vhc_velocity,
                               intruder_pts, intruder_velocity):
        """ Check if two vehicles are on a collision course.
         "Vehicle Collision Probability Calculation for General Traffic Scenarios Under Uncertainty",
         J. Ward, G. Agamennoni, S. Worrall, E. Nebot"""
        is_collision_path = False
        num_of_ego_pts = len(ego_vhc_pts)
        num_of_int_pts = len(intruder_pts)

        for i in range(num_of_ego_pts):
            # v_i_lin is a numpy array of linear velocity of the loom point 
            v_i_lin = ego_vhc_velocity + np.cross(ego_vhc_ang_velocity, np.subtract(ego_vhc_pts[i], ego_vhc_pos))
            min_ang = np.inf
            max_ang = -np.inf
            dist_of_min_pt = np.inf
            dist_of_max_pt = -np.inf
            for j in range(num_of_int_pts):
                distance_intruder_to_ego = np.subtract(ego_vhc_pts[i], intruder_pts[j])
                angle_intruder_to_ego = self.compute_ang_wrt_pos_lat_axis(distance_intruder_to_ego)
                if angle_intruder_to_ego < min_ang:
                    min_ang = angle_intruder_to_ego
                    dist_of_min_pt = distance_intruder_to_ego
                if angle_intruder_to_ego > max_ang:
                    max_ang = angle_intruder_to_ego
                    dist_of_max_pt = distance_intruder_to_ego

            if np.linalg.norm(dist_of_min_pt) != 0 and np.linalg.norm(dist_of_max_pt) != 0:
                loom_rate_of_min_pt = np.cross(dist_of_min_pt, v_i_lin) + np.cross(dist_of_min_pt, intruder_velocity)
                loom_rate_of_min_pt = loom_rate_of_min_pt / (np.linalg.norm(dist_of_min_pt) ** 2)
                loom_rate_of_max_pt = np.cross(dist_of_max_pt, v_i_lin) + np.cross(dist_of_max_pt, intruder_velocity)
                loom_rate_of_max_pt = loom_rate_of_max_pt / (np.linalg.norm(dist_of_max_pt) ** 2)
                if (loom_rate_of_min_pt[CoordinateSystem.LAT_AXIS] <= 0 <=
                        loom_rate_of_max_pt[CoordinateSystem.LAT_AXIS]):
                    is_collision_path = True
                    break
            else:
                is_collision_path = True
                break

        return is_collision_path

    def compute_ttc(self, pt_1, pt_2, vel_1, vel_2):
        """Computes the time to collision between points pt_1 and pt_2 given the velocities vel_1 and vel_2"""
        epsilon = 0.00001
        d = math.sqrt(np.dot(np.transpose(np.subtract(pt_1, pt_2)), np.subtract(pt_1, pt_2)))
        d_dot = np.dot(np.transpose(np.subtract(pt_1, pt_2)), np.subtract(vel_1, vel_2))[0] / d
        # d_2dot would be used for second-order TTC computation.
        # d_2dot = (np.dot(np.transpose(np.subtract(vel_1, vel_2)), np.subtract(vel_1, vel_2)) - d_dot ** 2) / d
        if abs(d_dot) > epsilon:
            ttc_1 = -d / d_dot
            if ttc_1 < 0:
                ttc_1 = None
        else:
            ttc_1 = None

        ttc = ttc_1
        return ttc

    def get_robustness(self):
        """Returns the minimum robustness (including any extra punishment)"""
        return self.minimum_robustness + self.extra_punishment

    def compute_robustness(self, current_sim_time_s):
        """Computes the Robustness value for current time"""
        if self.robustness_type == self.ROB_TYPE_DISTANCE_TO_PEDESTRIAN:
            self.extra_punishment = 0.0
            ego_pos = self.vehicles_manager.vehicles[0].current_position
            ego_pos[CoordinateSystem.VERT_AXIS] = 0.0
            for pedestrian in self.pedestrians_manager.pedestrians:
                ped_pos = pedestrian.current_position
                ped_pos[CoordinateSystem.VERT_AXIS] = 0.0
                dist_ego_ped = math.sqrt(np.dot(np.transpose(np.subtract(ped_pos, ego_pos)), np.subtract(ped_pos, ego_pos)))
                if dist_ego_ped < self.minimum_robustness:
                    self.minimum_robustness = dist_ego_ped
        else:
            self.minimum_robustness = self.MAX_ROBUSTNESS
            self.extra_punishment = 0.0
