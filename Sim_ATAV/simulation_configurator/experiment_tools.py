"""Tool functions that may be useful in different experiment setups.
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import math
import pickle
import random
import numpy as np
import pandas as pd
from Sim_ATAV.simulation_control.item_description import ItemDescription
from Sim_ATAV.simulation_control.webots_vehicle import WebotsVehicle
from Sim_ATAV.common.coordinate_system import CoordinateSystem
from Sim_ATAV.simulation_configurator import covering_array_utilities
from Sim_ATAV.vehicle_control.controller_commons.perception.sensor_fusion.sensor_fusion_tracker import ctrv_model
from Sim_ATAV.vehicle_control.controller_commons.path_following_tools import PathFollowingTools


RECT_OF_INTEREST_HALF_WIDTH = 0.85
VEHICLE_AXIS_TO_FRONT_LEN = 3.6
VEHICLE_AXIS_TO_REAR_LEN = 0.7
VEHICLE_HALF_WIDTH = 0.85
PEDESTRIAN_WIDTH = 0.15
path_following_tools = None


class TrajectoryExtensionData(object):
    EGO_TO_OBJECT_DISTANCE = 0
    EGO_TO_OBJECT_FUTURE_DISTANCE = 1
    EGO_TO_OBJECT_FUTURE_MIN_DISTANCE_TIME = 2

    def __init__(self, data, related_item):
        self.data = data
        self.related_item = related_item


def plot_future_trajectories(ego_future_pos_angle_list, agent_future_pos_angle_list):
    import matplotlib.pyplot as plt

    plt.plot(-ego_future_pos_angle_list[:, 0], ego_future_pos_angle_list[:, 1], 'g')
    plt.plot(-ego_future_pos_angle_list[0, 0], ego_future_pos_angle_list[0, 1], 'g*')
    agent_future_pos_angles = agent_future_pos_angle_list[0]
    plt.plot(-agent_future_pos_angles[:, 0], agent_future_pos_angles[:, 1], 'b')
    plt.plot(-agent_future_pos_angles[0, 0], agent_future_pos_angles[0, 1], 'b*')
    agent_future_pos_angles = agent_future_pos_angle_list[1]
    plt.plot(-agent_future_pos_angles[:, 0], agent_future_pos_angles[:, 1], 'k')
    plt.plot(-agent_future_pos_angles[0, 0], agent_future_pos_angles[0, 1], 'k*')
    plt.axis('equal')
    plt.show()


def extend_trajectory_with_requested_data(trajectory, sim_environment, list_of_new_data):
    traj_dict = sim_environment.simulation_trace_dict
    original_traj_width = trajectory.shape[1]
    new_trajectory = np.append(trajectory, np.zeros([len(trajectory), len(list_of_new_data)]), 1)
    for (traj_ind, traj_point) in enumerate(new_trajectory):
        future_distance_data_agent_states = []
        future_distance_data_traj_indices = []
        future_distance_min_time_data_traj_indices = []
        for (new_data_ind, new_data_item) in enumerate(list_of_new_data):
            if (new_data_item.data == TrajectoryExtensionData.EGO_TO_OBJECT_DISTANCE and
                    new_data_item.related_item.item_type == ItemDescription.ITEM_TYPE_VEHICLE):
                agent_vhc = sim_environment.agent_vehicles_list[new_data_item.related_item.item_index - 1]
                # -1, because index 0 is ego vhc and it is not on this list.
                new_trajectory[traj_ind][original_traj_width + new_data_ind] = distance_between_vehicles(
                    self_vhc_pos=np.array([traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, 0,
                                                                 WebotsVehicle.STATE_ID_POSITION_X)]],
                                           traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, 0,
                                                                 WebotsVehicle.STATE_ID_POSITION_Y)]]]),
                    self_vhc_orientation=traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, 0,
                                                               WebotsVehicle.STATE_ID_ORIENTATION)]],
                    self_vhc_front_length=VEHICLE_AXIS_TO_FRONT_LEN,
                    self_vhc_rear_length=VEHICLE_AXIS_TO_REAR_LEN,
                    self_vhc_width=VEHICLE_HALF_WIDTH,
                    ext_vhc_pos=np.array([traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE,
                                                                new_data_item.related_item.item_index,
                                                                WebotsVehicle.STATE_ID_POSITION_X)]],
                                          traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE,
                                                                new_data_item.related_item.item_index,
                                                                WebotsVehicle.STATE_ID_POSITION_Y)]]]),
                    ext_vhc_orientation=traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE,
                                                              new_data_item.related_item.item_index,
                                                              WebotsVehicle.STATE_ID_ORIENTATION)]],
                    ext_vhc_width=agent_vhc.half_width,
                    ext_vhc_rear_length=agent_vhc.rear_axis_to_rear_length,
                    ext_vhc_front_length=agent_vhc.rear_axis_to_front_length)
            elif (new_data_item.data == TrajectoryExtensionData.EGO_TO_OBJECT_FUTURE_DISTANCE and
                  new_data_item.related_item.item_type == ItemDescription.ITEM_TYPE_VEHICLE):
                future_distance_data_traj_indices.append(original_traj_width + new_data_ind)
                future_distance_data_agent_states.append(
                    [traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE,
                                           new_data_item.related_item.item_index,
                                           WebotsVehicle.STATE_ID_POSITION_X)]],
                     traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE,
                                           new_data_item.related_item.item_index,
                                           WebotsVehicle.STATE_ID_POSITION_Y)]],
                     traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE,
                                           new_data_item.related_item.item_index,
                                           WebotsVehicle.STATE_ID_SPEED)]],
                     traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE,
                                           new_data_item.related_item.item_index,
                                           WebotsVehicle.STATE_ID_ORIENTATION)]],
                     traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE,
                                           new_data_item.related_item.item_index,
                                           WebotsVehicle.STATE_ID_YAW_RATE)]]])
            elif (new_data_item.data == TrajectoryExtensionData.EGO_TO_OBJECT_FUTURE_MIN_DISTANCE_TIME and
                  new_data_item.related_item.item_type == ItemDescription.ITEM_TYPE_VEHICLE):
                # If we want the time for the minimum distance, number of EGO_TO_OBJECT_FUTURE_MIN_DISTANCE_TIME
                # requests must match the number of EGO_TO_OBJECT_FUTURE_DISTANCE requests.
                future_distance_min_time_data_traj_indices.append(original_traj_width + new_data_ind)
        if future_distance_data_agent_states:
            ego_states = \
                [traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, 0, WebotsVehicle.STATE_ID_POSITION_X)]],
                 traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, 0, WebotsVehicle.STATE_ID_POSITION_Y)]],
                 traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, 0, WebotsVehicle.STATE_ID_SPEED)]],
                 traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, 0, WebotsVehicle.STATE_ID_ORIENTATION)]],
                 traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, 0, WebotsVehicle.STATE_ID_YAW_RATE)]]]
            (min_dist_per_vhc, min_dist_time_per_vhc) = get_minimum_distance_in_future(
                ego_states=ego_states,
                agent_vehicles_states_list=future_distance_data_agent_states,
                target_points_list=sim_environment.ego_target_path,
                agent_vhc_list=sim_environment.agent_vehicles_list,
                maximum_time=min(3.5, float(int((ego_states[2]/3.0) / 0.1) * 0.1)),
                time_step=0.1,
                compute_threshold=70.0)  # Make maximum_time a multiple of time step.
            # Considering 8 m/s^2 as the emergency brake power, we check (speed / 2) time in the future.
            for (dist_ind, data_traj_ind) in enumerate(future_distance_data_traj_indices):
                new_trajectory[traj_ind][data_traj_ind] = min_dist_per_vhc[dist_ind]
            for (dist_ind, data_traj_ind) in enumerate(future_distance_min_time_data_traj_indices):
                new_trajectory[traj_ind][data_traj_ind] = min_dist_time_per_vhc[dist_ind]

    return new_trajectory


def npArray2Matlab(x):
    return x.tolist()


def rotate_point_ccw(point, theta):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.dot(np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]), point)


def rotate_rectangle_ccw(rect_corners, theta):
    new_rectangle = rect_corners[:]
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    for (ind, corner) in rect_corners:
        new_rectangle[ind] = np.dot(np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]), corner)
    return new_rectangle


def get_rectangle_of_interest(rect_corners, vhc_pos, vhc_length, vhc_orientation):
    new_rectangle = rotate_rectangle_ccw(rect_corners, vhc_orientation)
    new_rectangle = new_rectangle + [vhc_pos[CoordinateSystem.X_AXIS], vhc_pos[CoordinateSystem.Y_AXIS] + vhc_length]
    return new_rectangle


def is_pedestrian_in_rectangle_of_interest(vhc_pos, vhc_orientation, ped_pos, vhc_length, rect_half_width, rect_depth):
    ped_rotated = rotate_point_ccw(ped_pos-vhc_pos, -vhc_orientation)
    if -rect_half_width < ped_rotated[0] < rect_half_width and vhc_length < ped_rotated[1] < vhc_length + rect_depth:
        in_rectangle = True
        distance = ped_rotated[1] - vhc_length
    else:
        in_rectangle = False
        distance = abs(ped_rotated[1] - vhc_length) + rect_depth

    return in_rectangle, distance


def is_pedestrian_in_front_corridor(self_vhc_pos, self_vhc_orientation, self_vhc_length, self_vhc_width, point_pos):
    """Only in 2-D space (no z-axis in positions)"""
    # We consider pedestrians as circles with radius PEDESTRIAN_WIDTH
    point_in_vhc_coord = \
        rotate_point_ccw(np.array(point_pos) - np.array(self_vhc_pos), -1.0 * np.array(-self_vhc_orientation))
    vhc_left_front_corner_in_vhc_coord = [-self_vhc_width, self_vhc_length]
    vhc_right_front_corner_in_vhc_coord = [self_vhc_width, self_vhc_length]
    (temp_dist, t_ped) = line_dist(vhc_left_front_corner_in_vhc_coord, vhc_right_front_corner_in_vhc_coord,
                                   [point_in_vhc_coord[0], point_in_vhc_coord[1]])

    if (-self_vhc_width - PEDESTRIAN_WIDTH < point_in_vhc_coord[0] < self_vhc_width + PEDESTRIAN_WIDTH and
            point_in_vhc_coord[1] > self_vhc_length - PEDESTRIAN_WIDTH):
        in_corridor = True
        distance = max(0.0, temp_dist - PEDESTRIAN_WIDTH)
    else:
        in_corridor = False
        # Shortest distance to collision (when the point is outside the car's corridor):
        distance = max(0.01, temp_dist - PEDESTRIAN_WIDTH)
    return in_corridor, distance


def point_dist(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def line_dist(line_pt1, line_pt2, ref_point):
    """Computes distance from a point to a line segment."""
    # Modified from a code I have found on stackoverflow.
    line_len_squared = (line_pt2[0] - line_pt1[0])**2 + (line_pt2[1] - line_pt1[1])**2
    if line_len_squared == 0.0:
        dist = point_dist(line_pt1, ref_point)
        t = 0.0
        # closest_point = line_pt1[:]
    else:
        t = max(0.0, min(1.0, np.dot(np.array(ref_point) - np.array(line_pt1),
                                     np.array(line_pt2) - np.array(line_pt1)) / line_len_squared))
        # t is 0 to 1. 1 means closest point is pt2, 0 means closest point it pt1
        projection = np.array(line_pt1) + t*(np.array(line_pt2) - np.array(line_pt1))
        dist = point_dist(ref_point, projection)
        # closest_point = list(projection)
    return dist, t


def is_vehicle_in_front_corridor(self_vhc_pos, self_vhc_orientation, self_vhc_length, self_vhc_width, ext_vhc_pos,
                                 ext_vhc_orientation, ext_vhc_width, ext_vhc_rear_length, ext_vhc_front_length):
    """Only in 2-D space (no z-axis in positions)"""
    ext_vhc_frnt_left = rotate_point_ccw([-ext_vhc_width, ext_vhc_front_length], ext_vhc_orientation) + ext_vhc_pos
    ext_vhc_frnt_right = rotate_point_ccw([ext_vhc_width, ext_vhc_front_length], ext_vhc_orientation) + ext_vhc_pos
    ext_vhc_rear_left = rotate_point_ccw([-ext_vhc_width, -ext_vhc_rear_length], ext_vhc_orientation) + ext_vhc_pos
    ext_vhc_rear_right = rotate_point_ccw([ext_vhc_width, -ext_vhc_rear_length], ext_vhc_orientation) + ext_vhc_pos

    ext_vhc_frnt_left_in_vhc_coord = rotate_point_ccw(ext_vhc_frnt_left - self_vhc_pos, self_vhc_orientation)
    ext_vhc_frnt_right_in_vhc_coord = rotate_point_ccw(ext_vhc_frnt_right - self_vhc_pos, self_vhc_orientation)
    ext_vhc_rear_left_in_vhc_coord = rotate_point_ccw(ext_vhc_rear_left - self_vhc_pos, self_vhc_orientation)
    ext_vhc_rear_right_in_vhc_coord = rotate_point_ccw(ext_vhc_rear_right - self_vhc_pos, self_vhc_orientation)

    min_left = min(ext_vhc_frnt_left_in_vhc_coord[0],
                   ext_vhc_frnt_right_in_vhc_coord[0],
                   ext_vhc_rear_left_in_vhc_coord[0],
                   ext_vhc_rear_right_in_vhc_coord[0])
    max_right = max(ext_vhc_frnt_left_in_vhc_coord[0],
                    ext_vhc_frnt_right_in_vhc_coord[0],
                    ext_vhc_rear_left_in_vhc_coord[0],
                    ext_vhc_rear_right_in_vhc_coord[0])
    min_front = min(ext_vhc_frnt_left_in_vhc_coord[1],
                    ext_vhc_frnt_right_in_vhc_coord[1],
                    ext_vhc_rear_left_in_vhc_coord[1],
                    ext_vhc_rear_right_in_vhc_coord[1])
    max_front = max(ext_vhc_frnt_left_in_vhc_coord[1],
                    ext_vhc_frnt_right_in_vhc_coord[1],
                    ext_vhc_rear_left_in_vhc_coord[1],
                    ext_vhc_rear_right_in_vhc_coord[1])

    # Distance computation is just a best effort computation. Not exact.
    if (((-self_vhc_width < min_left < self_vhc_width or -self_vhc_width < max_right < self_vhc_width) or
         (min_left < -self_vhc_width and max_right > self_vhc_width)) and max_front > self_vhc_length):
        in_corridor = True

        (temp_dist, t1l) = line_dist(ext_vhc_frnt_left_in_vhc_coord, ext_vhc_frnt_right_in_vhc_coord,
                                     [-self_vhc_width, self_vhc_length])
        distance = temp_dist
        (temp_dist, t1r) = line_dist(ext_vhc_frnt_left_in_vhc_coord, ext_vhc_frnt_right_in_vhc_coord,
                                     [self_vhc_width, self_vhc_length])
        distance = min(distance, temp_dist)
        (temp_dist, t2l) = line_dist(ext_vhc_frnt_right_in_vhc_coord, ext_vhc_rear_right_in_vhc_coord,
                                     [-self_vhc_width, self_vhc_length])
        distance = min(distance, temp_dist)
        (temp_dist, t2r) = line_dist(ext_vhc_frnt_right_in_vhc_coord, ext_vhc_rear_right_in_vhc_coord,
                                     [self_vhc_width, self_vhc_length])
        distance = min(distance, temp_dist)
        (temp_dist, t3l) = line_dist(ext_vhc_rear_right_in_vhc_coord, ext_vhc_rear_left_in_vhc_coord,
                                     [-self_vhc_width, self_vhc_length])
        distance = min(distance, temp_dist)
        (temp_dist, t3r) = line_dist(ext_vhc_rear_right_in_vhc_coord, ext_vhc_rear_left_in_vhc_coord,
                                     [self_vhc_width, self_vhc_length])
        distance = min(distance, temp_dist)
        (temp_dist, t4l) = line_dist(ext_vhc_rear_left_in_vhc_coord, ext_vhc_frnt_left_in_vhc_coord,
                                     [-self_vhc_width, self_vhc_length])
        distance = min(distance, temp_dist)
        (temp_dist, t4r) = line_dist(ext_vhc_rear_left_in_vhc_coord, ext_vhc_frnt_left_in_vhc_coord,
                                     [self_vhc_width, self_vhc_length])
        distance = min(distance, temp_dist)
        distance = max(0.0, distance)
        if (distance < 1.0 and
                ((0.01 < t1l < 0.99 and 0.01 < t2l < 0.99 and 0.01 < t3l < 0.99 and 0.01 < t4l < 0.99) or
                 (0.01 < t1r < 0.99 and 0.01 < t2r < 0.99 and 0.01 < t3r < 0.99 and 0.01 < t4r < 0.99))):
            distance = 0.0  # Inside the rectangle
    else:
        in_corridor = False
        # Shortest distance to collision (when the point is outside the car's corridor):
        # point distance instead of line distance to reduce computation time.
        distance = point_dist(ext_vhc_frnt_left_in_vhc_coord, [-self_vhc_width, self_vhc_length])
        distance = min(distance, point_dist(ext_vhc_frnt_left_in_vhc_coord, [self_vhc_width, self_vhc_length]))
        distance = min(distance, point_dist(ext_vhc_frnt_left_in_vhc_coord, [-self_vhc_width, self_vhc_length]))
        distance = min(distance, point_dist(ext_vhc_frnt_right_in_vhc_coord, [self_vhc_width, self_vhc_length]))
        distance = min(distance, point_dist(ext_vhc_rear_right_in_vhc_coord, [-self_vhc_width, self_vhc_length]))
        distance = min(distance, point_dist(ext_vhc_rear_right_in_vhc_coord, [self_vhc_width, self_vhc_length]))
        distance = min(distance, point_dist(ext_vhc_rear_left_in_vhc_coord, [-self_vhc_width, self_vhc_length]))
        distance = min(distance, point_dist(ext_vhc_rear_left_in_vhc_coord, [self_vhc_width, self_vhc_length]))
        distance = max(0.01, distance)
    return in_corridor, distance


def get_minimum_distance_in_future(ego_states, agent_vehicles_states_list, target_points_list, agent_vhc_list,
                                   maximum_time=5.0, time_step=0.1, compute_threshold=math.inf, debug_plot=False):
    global path_following_tools

    if path_following_tools is None:
        path_following_tools = PathFollowingTools(target_points=target_points_list)

    cur_position = ego_states[0:2]
    cur_speed_m_s = ego_states[2]
    cur_yaw_angle = ego_states[3]
    agent_future_pos_angle_list = []
    ego_future_pos_angle_list = np.empty((0, 3), dtype=float)
    distances = []
    for (agent_vhc_ind, agent_vhc_cur_state) in enumerate(agent_vehicles_states_list):
        distances.append(np.linalg.norm(np.array([cur_position[0] - agent_vhc_cur_state[0],
                                                       cur_position[1] - agent_vhc_cur_state[1]])))
    if any(item < compute_threshold for item in distances):
        for time_val in np.linspace(time_step, maximum_time, int(maximum_time / time_step), endpoint=True):
            (last_segment_ind, line_segment_as_list, nearest_pos_on_path, dist_to_end_of_segment) = \
                path_following_tools.get_current_segment(vhc_pos=cur_position)
            (self_future_pos, self_future_angle) = path_following_tools.get_expected_position_angle_at_time(
                target_time=time_val, current_position=cur_position, current_speed_m_s=cur_speed_m_s,
                current_angle=cur_yaw_angle, current_segment_ind=last_segment_ind)
            ego_future_pos_angle_list = np.append(ego_future_pos_angle_list,
                                                  np.array([[self_future_pos[0],
                                                             self_future_pos[1],
                                                             self_future_angle]]), axis=0)

    min_dist_per_vhc = [math.inf] * len(agent_vehicles_states_list)
    min_dist_time_per_vhc = [maximum_time] * len(agent_vehicles_states_list)
    for (agent_vhc_ind, agent_vhc_cur_state) in enumerate(agent_vehicles_states_list):
        agent_vhc = agent_vhc_list[agent_vhc_ind]
        temp_agent_state = agent_vhc_cur_state[:]
        min_dist = math.inf
        min_dist_time = maximum_time
        agent_future_pos_angles = np.empty((0, 3), dtype=float)
        if distances[agent_vhc_ind] < compute_threshold:
            for (time_ind, time_val) in enumerate(np.linspace(time_step, maximum_time, int(maximum_time / time_step),
                                                              endpoint=True)):
                temp_agent_state = ctrv_model(state=temp_agent_state, delta_t=time_step)
                temp_dist = distance_between_vehicles(self_vhc_pos=ego_future_pos_angle_list[time_ind][0:2],
                                                      self_vhc_orientation=ego_future_pos_angle_list[time_ind][2],
                                                      self_vhc_front_length=VEHICLE_AXIS_TO_FRONT_LEN,
                                                      self_vhc_rear_length=VEHICLE_AXIS_TO_REAR_LEN,
                                                      self_vhc_width=RECT_OF_INTEREST_HALF_WIDTH,
                                                      ext_vhc_pos=temp_agent_state[0:2],
                                                      ext_vhc_orientation=temp_agent_state[3],
                                                      ext_vhc_width=agent_vhc.half_width,
                                                      ext_vhc_rear_length=agent_vhc.rear_axis_to_rear_length,
                                                      ext_vhc_front_length=agent_vhc.rear_axis_to_front_length)
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    min_dist_time = time_val
                if debug_plot:
                    agent_future_pos_angles = np.append(agent_future_pos_angles,
                                                        np.array([[temp_agent_state[0],
                                                                   temp_agent_state[1],
                                                                   temp_agent_state[3]]]), axis=0)
        else:
            min_dist = distances[agent_vhc_ind]
        min_dist_per_vhc[agent_vhc_ind] = min_dist
        min_dist_time_per_vhc[agent_vhc_ind] = min_dist_time
        if debug_plot:
            agent_future_pos_angle_list.append(agent_future_pos_angles)
    if debug_plot:
        plot_future_trajectories(ego_future_pos_angle_list, agent_future_pos_angle_list)
    return min_dist_per_vhc, min_dist_time_per_vhc


def extend_trajectory(trajectory, traj_dict, vhc_list, is_correcting=False, is_compute_det_perf=True, num_ped=1,
                      num_vhc=5):
    if is_correcting:
        new_trajectory = trajectory
    else:
        new_trajectory = np.append(trajectory, np.zeros([len(trajectory), 2*(num_ped + num_vhc)]), 1)
    min_dist = 10000.0
    critical_item_type = 'Pedestrian'
    critical_item_ind = 0
    minimum_ind = 0
    for (traj_ind, traj_point) in enumerate(new_trajectory):
        self_vhc_pos = \
            np.array([traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, 0, WebotsVehicle.STATE_ID_POSITION_X)]],
                      traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, 0, WebotsVehicle.STATE_ID_POSITION_Y)]]])
        self_vhc_orientation = \
            traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, 0, WebotsVehicle.STATE_ID_ORIENTATION)]]
        for ped_ind in range(num_ped):
            ped_pos = np.array([traj_point[traj_dict[(ItemDescription.ITEM_TYPE_PEDESTRIAN,
                                                      ped_ind,
                                                      WebotsVehicle.STATE_ID_POSITION_X)]],
                                traj_point[traj_dict[(ItemDescription.ITEM_TYPE_PEDESTRIAN,
                                                      ped_ind,
                                                      WebotsVehicle.STATE_ID_POSITION_Y)]]])
            (in_corridor, distance) = is_pedestrian_in_front_corridor(self_vhc_pos,
                                                                      self_vhc_orientation,
                                                                      VEHICLE_AXIS_TO_FRONT_LEN,
                                                                      RECT_OF_INTEREST_HALF_WIDTH,
                                                                      ped_pos)
            traj_offset = -2*(num_vhc + num_ped - ped_ind)
            new_trajectory[traj_ind][traj_offset] = in_corridor
            new_trajectory[traj_ind][traj_offset + 1] = distance
            if in_corridor and distance < min_dist:
                min_dist = distance
                critical_item_type = 'Pedestrian'
                critical_item_ind = ped_ind
                minimum_ind = traj_ind
        for vhc_ind in range(1, num_vhc+1):  # Taking care of the fact that the ego vehicle is the one indexed as 0
            agent_vhc = vhc_list[vhc_ind-1]
            ext_vhc_pos = np.array([traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE,
                                                          vhc_ind,
                                                          WebotsVehicle.STATE_ID_POSITION_X)]],
                                    traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE,
                                                          vhc_ind,
                                                          WebotsVehicle.STATE_ID_POSITION_Y)]]])
            ext_vhc_orientation = traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE,
                                                        vhc_ind,
                                                        WebotsVehicle.STATE_ID_ORIENTATION)]]
            # TODO: For better results:
            # check the model of the corresponding vehicle and use that model's dimensions instead of fixed values.
            (in_corridor, distance) = is_vehicle_in_front_corridor(self_vhc_pos,
                                                                   self_vhc_orientation,
                                                                   VEHICLE_AXIS_TO_FRONT_LEN,
                                                                   RECT_OF_INTEREST_HALF_WIDTH,
                                                                   ext_vhc_pos,
                                                                   ext_vhc_orientation,
                                                                   agent_vhc.half_width,
                                                                   agent_vhc.rear_axis_to_rear_length,
                                                                   agent_vhc.rear_axis_to_front_length)
            traj_offset = -2*(num_vhc - vhc_ind + 1)
            new_trajectory[traj_ind][traj_offset] = in_corridor
            new_trajectory[traj_ind][traj_offset + 1] = distance
            if in_corridor and distance < min_dist:
                min_dist = distance
                critical_item_type = 'Car'
                critical_item_ind = vhc_ind
                minimum_ind = traj_ind
    traj_point = new_trajectory[minimum_ind]
    if is_compute_det_perf:
        if critical_item_type == 'Pedestrian':
            det_perf = traj_point[traj_dict[(ItemDescription.ITEM_TYPE_PED_DET_PERF, 0, critical_item_ind + 1)]]
        else:
            det_perf = traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE_DET_PERF, 0, critical_item_ind + 1)]]
    else:
        det_perf = 0.0
    print('In trajectory: Minimum distance: {:.2f}, item type: {} item index: {} time index: {} Det. Perf: {:.2f}'.
          format(min_dist, critical_item_type, critical_item_ind, minimum_ind, det_perf))
    return new_trajectory, det_perf


def distance_between_vehicles(self_vhc_pos, self_vhc_orientation, self_vhc_front_length, self_vhc_rear_length,
                              self_vhc_width, ext_vhc_pos, ext_vhc_orientation, ext_vhc_width, ext_vhc_rear_length,
                              ext_vhc_front_length):
    """Only in 2-D space (no z-axis in positions)"""
    ext_vhc_frnt_left = rotate_point_ccw([-ext_vhc_width, ext_vhc_front_length], -ext_vhc_orientation) + ext_vhc_pos
    ext_vhc_frnt_right = rotate_point_ccw([ext_vhc_width, ext_vhc_front_length], -ext_vhc_orientation) + ext_vhc_pos
    ext_vhc_rear_left = rotate_point_ccw([-ext_vhc_width, -ext_vhc_rear_length], -ext_vhc_orientation) + ext_vhc_pos
    ext_vhc_rear_right = rotate_point_ccw([ext_vhc_width, -ext_vhc_rear_length], -ext_vhc_orientation) + ext_vhc_pos

    ext_vhc_frnt_left_in_vhc_coord = rotate_point_ccw(ext_vhc_frnt_left - self_vhc_pos, -self_vhc_orientation)
    ext_vhc_frnt_right_in_vhc_coord = rotate_point_ccw(ext_vhc_frnt_right - self_vhc_pos, -self_vhc_orientation)
    ext_vhc_rear_left_in_vhc_coord = rotate_point_ccw(ext_vhc_rear_left - self_vhc_pos, -self_vhc_orientation)
    ext_vhc_rear_right_in_vhc_coord = rotate_point_ccw(ext_vhc_rear_right - self_vhc_pos, -self_vhc_orientation)

    ext_vehicle_lines = [[ext_vhc_frnt_left_in_vhc_coord, ext_vhc_frnt_right_in_vhc_coord],
                         [ext_vhc_frnt_right_in_vhc_coord, ext_vhc_rear_right_in_vhc_coord],
                         [ext_vhc_rear_right_in_vhc_coord, ext_vhc_rear_left_in_vhc_coord],
                         [ext_vhc_rear_left_in_vhc_coord, ext_vhc_frnt_left_in_vhc_coord]]

    ext_vehicle_corners = [ext_vhc_frnt_left_in_vhc_coord, ext_vhc_frnt_right_in_vhc_coord,
                           ext_vhc_rear_right_in_vhc_coord, ext_vhc_rear_left_in_vhc_coord]

    ego_points = [np.array([-self_vhc_width, self_vhc_front_length]),
                  np.array([self_vhc_width, self_vhc_front_length]),
                  np.array([-self_vhc_width, self_vhc_rear_length]),
                  np.array([self_vhc_width, self_vhc_rear_length])]

    distance = math.inf
    # Compute the minimum distance from each corner of the external vehicle to the edges of the ego vehicle:
    # This is easier because the external vehicle is already represented in the ego vehicle's coordinate system.
    for ext_vehicle_corner in ext_vehicle_corners:
        if -self_vhc_width < ext_vehicle_corner[0] < self_vhc_width:
            x_dist = 0.0
        elif ext_vehicle_corner[0] > self_vhc_width:
            x_dist = ext_vehicle_corner[0] - self_vhc_width
        else:
            x_dist = -self_vhc_width - ext_vehicle_corner[0]
        if -self_vhc_rear_length < ext_vehicle_corner[1] < self_vhc_front_length:
            y_dist = 0.0
        elif ext_vehicle_corner[1] > self_vhc_front_length:
            y_dist = ext_vehicle_corner[1] - self_vhc_front_length
        else:
            y_dist = -self_vhc_rear_length - ext_vehicle_corner[1]
        temp_dist = math.sqrt(x_dist**2 + y_dist**2)
        distance = min(distance, temp_dist)

    # Compute the minimum distance from each corner of the ego vehicle to the edges of the external vehicle:
    for ego_point in ego_points:
        num_inside_pts = 0
        for ext_vehicle_line in ext_vehicle_lines:
            (temp_dist, t) = line_dist(ext_vehicle_line[0], ext_vehicle_line[1], ego_point)
            if 0.0001 < t < 0.9999:  # NOT (on a line or outside one of the lines).
                # When the closest point on the line is one end of the line (t==0 or t==1),
                # then the point is outside the rectangle.
                num_inside_pts += 1
            else:
                distance = min(distance, temp_dist)
        if num_inside_pts == len(ext_vehicle_lines):
            distance = 0.0
        if distance == 0.0:
            break
    return distance


def extend_trajectory_with_distance(trajectory, traj_dict, vhc_list, num_ped=0, num_vhc=2):
    new_trajectory = np.append(trajectory, np.zeros([len(trajectory), (num_ped + num_vhc)]), 1)
    min_dist = 10000.0
    critical_item_type = 'Pedestrian'
    critical_item_ind = 0
    minimum_ind = 0
    for (traj_ind, traj_point) in enumerate(new_trajectory):
        self_vhc_pos = \
            np.array([traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, 0, WebotsVehicle.STATE_ID_POSITION_X)]],
                      traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, 0, WebotsVehicle.STATE_ID_POSITION_Y)]]])
        self_vhc_orientation = \
            traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE, 0, WebotsVehicle.STATE_ID_ORIENTATION)]]
        for ped_ind in range(num_ped):
            ped_pos = np.array([traj_point[traj_dict[(ItemDescription.ITEM_TYPE_PEDESTRIAN,
                                                      ped_ind,
                                                      WebotsVehicle.STATE_ID_POSITION_X)]],
                                traj_point[traj_dict[(ItemDescription.ITEM_TYPE_PEDESTRIAN,
                                                      ped_ind,
                                                      WebotsVehicle.STATE_ID_POSITION_Y)]]])
            (in_corridor, distance) = is_pedestrian_in_front_corridor(self_vhc_pos,
                                                                      self_vhc_orientation,
                                                                      VEHICLE_AXIS_TO_FRONT_LEN,
                                                                      RECT_OF_INTEREST_HALF_WIDTH,
                                                                      ped_pos)
            traj_offset = -1*(num_vhc + num_ped - ped_ind)
            new_trajectory[traj_ind][traj_offset] = distance
            if distance < min_dist:
                min_dist = distance
                critical_item_type = 'Pedestrian'
                critical_item_ind = ped_ind
                minimum_ind = traj_ind
        for vhc_ind in range(1, num_vhc+1):  # Taking care of the fact that the ego vehicle is the one indexed as 0
            agent_vhc = vhc_list[vhc_ind-1]
            ext_vhc_pos = np.array([traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE,
                                                          vhc_ind,
                                                          WebotsVehicle.STATE_ID_POSITION_X)]],
                                    traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE,
                                                          vhc_ind,
                                                          WebotsVehicle.STATE_ID_POSITION_Y)]]])
            ext_vhc_orientation = traj_point[traj_dict[(ItemDescription.ITEM_TYPE_VEHICLE,
                                                        vhc_ind,
                                                        WebotsVehicle.STATE_ID_ORIENTATION)]]
            # TODO: For better results:
            # check the model of the corresponding vehicle and use that model's dimensions instead of fixed values.
            distance = distance_between_vehicles(self_vhc_pos=self_vhc_pos,
                                                 self_vhc_orientation=self_vhc_orientation,
                                                 self_vhc_front_length=VEHICLE_AXIS_TO_FRONT_LEN,
                                                 self_vhc_rear_length=VEHICLE_AXIS_TO_REAR_LEN,
                                                 self_vhc_width=RECT_OF_INTEREST_HALF_WIDTH,
                                                 ext_vhc_pos=ext_vhc_pos,
                                                 ext_vhc_orientation=ext_vhc_orientation,
                                                 ext_vhc_width=agent_vhc.half_width,
                                                 ext_vhc_rear_length=agent_vhc.rear_axis_to_rear_length,
                                                 ext_vhc_front_length=agent_vhc.rear_axis_to_front_length)
            traj_offset = -1*(num_vhc - vhc_ind + 1)
            new_trajectory[traj_ind][traj_offset] = distance
            if distance < min_dist:
                min_dist = distance
                critical_item_type = 'Car'
                critical_item_ind = vhc_ind
                minimum_ind = traj_ind
    print('In trajectory: Minimum distance: {:.2f}, item type: {} item index: {} time index: {}'.
          format(min_dist, critical_item_type, critical_item_ind, minimum_ind))
    return new_trajectory


def save_robustness_values_to_csv(from_csv_file_name, to_csv_file_name, robustness_list):
    data_frame = covering_array_utilities.load_experiment_results_data(from_csv_file_name)
    data_frame = covering_array_utilities.add_column_to_data_frame(data_frame, 'robustness', 10000.0)
    for exp_ind in range(len(robustness_list)):
        covering_array_utilities.set_experiment_field_value(data_frame, exp_ind, 'robustness', robustness_list[exp_ind])
    covering_array_utilities.save_experiment_results(to_csv_file_name, data_frame)


def get_random_number_in_range(min_val=0.0, max_val=1.0, num_type=float):
    if num_type is float:
        random_num = random.uniform(min_val, max_val)
    else:  # int
        random_num = random.randint(int(min_val), int(max_val))
    return random_num


def save_random_number_generator_state(file_name):
    with open(file_name, 'wb+') as f:
        random_state = random.getstate()
        pickle.dump(random_state, f)
        f.close()


def restore_random_number_generator_state(file_name):
    with open(file_name, 'rb') as f:
        random_state = pickle.load(f)
        random.setstate(random_state)
        f.close()


def convert_side_enum_to_text(side_enum):
    if side_enum == 1:
        side_text = 'LEFT'
    else:
        side_text = 'RIGHT'
    return side_text


def load_environment_configuration(exp_file_path, exp_file_name):
    environment_config_dict = {}
    try:
        environment_config_dict['env_config_file_name'] = exp_file_path + '/' + exp_file_name + '_env_config.csv'
        env_config_data_frame = pd.read_csv(environment_config_dict['env_config_file_name'], index_col=0)
        env_config_fields = env_config_data_frame.iloc[[0]]
        environment_config_dict['exp_short_name'] = env_config_fields['exp_short_name'].iloc[0]
        if env_config_fields['exp_config_folder_is_absolute'].iloc[0]:
            environment_config_dict['exp_config_folder'] = env_config_fields['exp_config_folder'].iloc[0] + '/'
        else:
            environment_config_dict['exp_config_folder'] = \
                exp_file_path + '/' + env_config_fields['exp_config_folder'].iloc[0] + '/'

        if env_config_fields['world_file_folder_is_absolute'].iloc[0]:
            environment_config_dict['world_file_path'] = env_config_fields['world_file_folder'].iloc[0] + '/'
        else:
            environment_config_dict['world_file_path'] = \
                exp_file_path + '/' + env_config_fields['world_file_folder'].iloc[0] + '/'
        environment_config_dict['world_file_name'] = env_config_fields['world_file_name'].iloc[0]
        environment_config_dict['is_save_trajectory_files'] = env_config_fields['is_save_trajectory_files'].iloc[0]
        if env_config_fields['traj_log_folder_is_absolute'].iloc[0]:
            environment_config_dict['trajectory_log_folder'] = env_config_fields['traj_log_folder'].iloc[0] + '/'
        else:
            environment_config_dict['trajectory_log_folder'] = \
                exp_file_path + '/' + env_config_fields['traj_log_folder'].iloc[0] + '/'
        environment_config_dict['is_save_exp_results_file'] = env_config_fields['is_save_exp_results_file'].iloc[0]
        if env_config_fields['exp_results_folder_is_absolute'].iloc[0]:
            environment_config_dict['exp_results_folder'] = env_config_fields['exp_results_folder'].iloc[0] + '/'
        else:
            environment_config_dict['exp_results_folder'] = \
                exp_file_path + '/' + env_config_fields['exp_results_folder'].iloc[0] + '/'
    except Exception as ex:
        print('environment config could not be loaded. ERROR: ' + repr(ex))
    return environment_config_dict


def main():
    self_vhc_pos = [1.7838, 54.4649]
    self_vhc_orientation = 0.1771
    point_pos = [0.18, 58.0]
    self_vhc_width = 0.85
    self_vhc_length = 3.6
    is_pedestrian_in_front_corridor(self_vhc_pos, self_vhc_orientation, self_vhc_length, self_vhc_width, point_pos)


if __name__ == "__main__":
    main()
