"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import numpy as np
import math
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation.sensor_object import SensorObject
from Sim_ATAV.vehicle_control.controller_commons.visualization.console_output import ConsoleOutput
from Sim_ATAV.vehicle_control.controller_commons.controller_commons \
    import convert_global_to_relative_position, rotate_point_ccw


class TrajectoryEstimation(object):
    def __init__(self, console_output, path_following_tools):
        if console_output is None:
            self.console_output = ConsoleOutput(debug_mode=False)
        else:
            self.console_output = console_output
        self.path_following_tools = path_following_tools
        self.ego_future = []

    def estimate_ego_future(self, cur_position, cur_speed_ms, cur_yaw_angle, current_segment_ind):
        self.ego_future = []
        np_cur_pos = np.array(cur_position)
        for time_val in np.linspace(0.25, 5.0, 20, endpoint=True):
            (temp_abs_pos, _angle) = self.path_following_tools.get_expected_position_angle_at_time(
                time_val, cur_position, cur_speed_ms, current_segment_ind,
                target_path=self.path_following_tools.target_path)
            # convert to relative position to the current position. Because object positions are also relative.
            future_rel_pos = np.array(temp_abs_pos) - np_cur_pos
            future_rel_pos = \
                [math.cos(cur_yaw_angle) * future_rel_pos[0] + math.sin(cur_yaw_angle) * future_rel_pos[1],
                 -math.sin(cur_yaw_angle) * future_rel_pos[0] + math.cos(cur_yaw_angle) * future_rel_pos[1]]
            self.ego_future.append(future_rel_pos[:])
        return self.ego_future

    def estimate_future_ego_agent_conflicts(self, detected_objects, cur_ego_position, cur_ego_yaw_angle):
        for obj_ind in range(len(detected_objects)):
            if detected_objects[obj_ind].tracker is not None:
                obj_w = None
                vector_to_obj = np.array(detected_objects[obj_ind].object_position)
                # Check if object is in front cone and closer than 100 m.
                min_dist = np.linalg.norm(vector_to_obj)
                min_dist_time = 0.0
                min_lat_dist = vector_to_obj[0]
                min_long_dist = vector_to_obj[1]
                obj_projection = detected_objects[obj_ind].tracker.tracked_object_state[:]
                detected_objects[obj_ind].future = []
                if ((np.linalg.norm(vector_to_obj) < 100.0 and
                    abs(math.atan2(vector_to_obj[0], vector_to_obj[1])) < math.pi/4.0) or
                        np.linalg.norm(vector_to_obj) < 50.0 and
                        abs(math.atan2(vector_to_obj[0], vector_to_obj[1])) < math.pi/2.0):
                    for (time_ind, time_val) in enumerate(np.linspace(0.25, 5.0, 20, endpoint=True)):
                        # Both self_future_pos and obj_future_pos below are relative to current ego position.
                        self_future_pos = self.ego_future[time_ind]
                        # As the time_step granularity increases, the following will perform poorer.
                        obj_projection = detected_objects[obj_ind].tracker.get_projection_with_state(obj_projection,
                                                                                                     time_step=0.1,
                                                                                                     time_duration=0.25)
                        obj_future_pos = convert_global_to_relative_position(object_global_position=obj_projection[0:2],
                                                                             ego_global_position=cur_ego_position,
                                                                             ego_global_yaw_angle=cur_ego_yaw_angle)
                        detected_objects[obj_ind].future.append(obj_future_pos[:])
                        self.console_output.debug_print('time {} self_pos: {} obj_pos: {} obj_cur_pos: {}'.format(
                            time_ind, self_future_pos, obj_future_pos, detected_objects[obj_ind].object_position))
                        temp_vector = np.array(obj_future_pos) - np.array(self_future_pos)
                        lat_dist = temp_vector[0]
                        long_dist = temp_vector[1]
                        dist = np.linalg.norm(temp_vector)
                        if (dist < 5.0) and (detected_objects[obj_ind].object_type != SensorObject.OBJECT_PEDESTRIAN):
                            # Check all corners of the object.
                            positive_lat_found = False
                            negative_lat_found = False
                            corner_local_coords = [[-1.0, 2.0], [1.0, 2.0], [-1.0, -2.0], [1.0, -2.0]]
                            for local_coord in corner_local_coords:
                                rotated_local_coord = \
                                    rotate_point_ccw(point=np.transpose(np.array(local_coord)),
                                                     rotation_angle=-detected_objects[obj_ind].object_yaw_angle)
                                corner_future_pos = [obj_future_pos[0] + rotated_local_coord[0],
                                                     obj_future_pos[1] + rotated_local_coord[1]]
                                temp_vector = np.array(corner_future_pos) - np.array(self_future_pos)
                                temp_dist = np.linalg.norm(temp_vector)
                                positive_lat_found = positive_lat_found or (temp_vector[0] > 0)
                                negative_lat_found = negative_lat_found or (temp_vector[0] < 0)
                                if temp_dist < dist:  # this corner is closer
                                    dist = temp_dist
                                    lat_dist = temp_vector[0]
                                    long_dist = temp_vector[1]
                                    self.console_output.debug_print('min dist updated at corner. dist: {}'.format(dist))
                            if positive_lat_found and negative_lat_found:
                                lat_dist = 0.0  # A corner is on the right, other on the left/

                        if dist < min_dist:
                            min_dist = dist
                            if obj_w is None:  # Compute obj_w only when necessary
                                if SensorObject.SENSOR_LIDAR in detected_objects[obj_ind].sensor_aux_data_dict:
                                    cluster = detected_objects[obj_ind].sensor_aux_data_dict[
                                        SensorObject.SENSOR_LIDAR].lidar_cluster
                                    obj_w = abs(abs((cluster.min_x + cluster.max_x) / 2.0) - abs(
                                        min(cluster.cluster_points[:, 0],
                                            key=abs)))
                                    self.console_output.debug_print('obj_w = :{}'.format(obj_w))
                                else:
                                    obj_w = 0.8
                            # Not doing a great job here but this works okay.
                            min_lat_dist = min(lat_dist, abs(lat_dist - obj_w))
                            min_long_dist = long_dist
                            min_dist_time = time_val
                        elif dist > min_dist + 0.5:
                            break
                    detected_objects[obj_ind].aux_data['future_intersection'] = \
                        (min_dist, min_dist_time, min_lat_dist, min_long_dist)
                    self.console_output.debug_print(
                        'MINIMUM DISTANCE WITH OBJ :{} ({}) at time {} obj cur pos: {}'.format(
                            min_dist, [min_lat_dist, min_long_dist], min_dist_time,
                            detected_objects[obj_ind].object_position))
