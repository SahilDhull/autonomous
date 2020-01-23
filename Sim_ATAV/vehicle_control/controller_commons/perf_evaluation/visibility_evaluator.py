"""Defines VisibilityController class.
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import math
import numpy as np


class VisibilityConfig(object):
    OBJ_TYPE_IND = 0
    OBJ_ID_IND = 1

    def __init__(self, sensor, object_list, vehicle_id):
        self.sensor = sensor
        self.object_list = object_list[:]  # A tuple of type, id
        self.vehicle_id = vehicle_id

    def get_target_obj_info_as_dictionary_key(self, target_obj_ind):
        """When we use VisibilityConfig in the simulation trace, this is how we use it as a key in the
        trajectory dictionary for easy reference."""
        return (self.sensor.name, self.vehicle_id, self.object_list[target_obj_ind][self.OBJ_TYPE_IND],
                self.object_list[target_obj_ind][self.OBJ_ID_IND])


class VisibilityRatio(object):
    def __init__(self, percent, total_angle):
        self.percent = percent
        self.total_angle = total_angle


class VisibilitySensor(object):
    def __init__(self, sensor_name='Sensor', hor_fov=0.0, max_range=0.0, position=(0.0, 0.0, 0.0), x_rotation=0.0):
        self.name = sensor_name
        self.hor_fov = hor_fov
        self.max_range = max_range
        self.local_position = np.array([position[0], position[1], position[2]])
        self.local_position.shape = (3, 1)
        self.local_rotation = None
        self.x_rotation = x_rotation
        self.set_rotation_x_axis(x_rotation)

    def set_rotation_x_axis(self, x_rotation):
        self.local_rotation = np.array([[1.0, 0.0, 0.0],
                                        [0.0, math.cos(x_rotation), -math.sin(x_rotation)],
                                        [0.0, math.sin(x_rotation), math.cos(x_rotation)]])


class VisibilityLine(object):
    def __init__(self, left_angle, left_dist, right_angle, right_dist, object_id, object_type):
        self.left_angle = left_angle
        self.left_dist = left_dist
        self.right_angle = right_angle
        self.right_dist = right_dist
        self.object_id = object_id
        self.object_type = object_type

    def distance_at_angle(self, angle):
        ratio = (angle - self.left_angle) / (self.right_angle - self.left_angle) if self.right_angle != self.left_angle\
            else 0.0
        return self.left_dist + ratio * (self.right_dist - self.left_dist)


def convert_from_local_to_world_coordinates(object_rotation_matrix_3d,
                                            object_position,
                                            local_position):
    """Take the object position, local coordinates of a point in the
    object's coordinate system and the rotation matrix of the object.
    Return the position of the point in the world coordinate system."""
    return np.matmul(object_rotation_matrix_3d, local_position) + object_position


def convert_from_world_to_sensor_coordinates(sensor_rotation_matrix_3d,
                                             sensor_position,
                                             point_world_coordinates):
    """Consider the position and rotation of the sensor,
    and convert the world coordinates of the point to the sensor coordinate system,
    which is defined wrt the sensor position and rotation."""
    R = np.append(np.transpose(sensor_rotation_matrix_3d), np.zeros([3, 1]), axis=1)
    R = np.append(R, np.zeros([1, 4]), axis=0)
    R[3, 3] = 1.0

    C = np.eye(4)
    C[0, 3] = -sensor_position[0]
    C[1, 3] = -sensor_position[1]
    C[2, 3] = -sensor_position[2]
    W = np.append(point_world_coordinates, np.ones([1, 1]), axis=0)
    return np.matmul(R, np.matmul(C, W))


def get_object_lines_and_angles(obj_position, obj_rotation, obj_corners, sensor_position, sensor_rotation,
                                obj_id, obj_type):
    obj_pos = np.array(obj_position)
    obj_pos.shape = (3, 1)
    obj_rot = np.array(obj_rotation)
    obj_rot.shape = (3, 3)
    last_pt_dist = None
    last_pt_angle = None
    lines = []
    left_angle = math.inf
    right_angle = -math.inf
    for corner_id in [0, 2, 6, 4, 0]:  # corners are front-right bottom, top, f-l bottom, top, rear-r b,t, rear-left b,t
        local_coord = np.array(obj_corners[corner_id])
        local_coord.shape = (3, 1)
        world_coord = convert_from_local_to_world_coordinates(obj_rot, obj_pos, local_coord)
        sensor_coord = convert_from_world_to_sensor_coordinates(sensor_rotation, sensor_position, world_coord)
        pt_dist = math.sqrt(sensor_coord[0] ** 2 + sensor_coord[2] ** 2)
        pt_x_angle = math.atan2(-sensor_coord[0], sensor_coord[2])
        if pt_x_angle > math.pi:
            pt_x_angle -= 2*math.pi
        if pt_x_angle < -math.pi:
            pt_x_angle += 2*math.pi
        left_angle = min(left_angle, pt_x_angle)
        right_angle = max(right_angle, pt_x_angle)
        if right_angle - left_angle < math.pi:  # Otherwise, object is on the rear
            if last_pt_dist is not None:
                if last_pt_angle < pt_x_angle:
                    # new point is on the right
                    lines.append(VisibilityLine(left_angle=last_pt_angle, left_dist=last_pt_dist,
                                                right_angle=pt_x_angle, right_dist=pt_dist,
                                                object_id=obj_id, object_type=obj_type))
                else:
                    # new point is on the left
                    lines.append(VisibilityLine(left_angle=pt_x_angle, left_dist=pt_dist,
                                                right_angle=last_pt_angle, right_dist=last_pt_dist,
                                                object_id=obj_id, object_type=obj_type))
            last_pt_dist = pt_dist
            last_pt_angle = pt_x_angle
    return lines, left_angle, right_angle


def get_important_angles_from_lines(lines, left_angle, right_angle):
    important_angles = [left_angle, right_angle]

    for line in lines:
        if left_angle < line.left_angle < right_angle:
            important_angles.append(line.left_angle)
        if left_angle < line.right_angle < right_angle:
            important_angles.append(line.right_angle)
    important_angles = sorted(set(important_angles))  # sorted, unique values
    return important_angles


def get_closest_line_on_a_direction(angle, lines):
    closest_line_index = None
    closest_line_dist = math.inf
    epsilon = math.pi/36000.0

    for (line_ind, line) in enumerate(lines):
        if line.left_angle - epsilon <= angle <= line.right_angle + epsilon:
            dist = line.distance_at_angle(angle)
            if dist < closest_line_dist:
                closest_line_index = line_ind
                closest_line_dist = dist
    return closest_line_index, closest_line_dist


def compute_visibilities(important_angles, all_lines, obj_angles_dict, self_vhc_id, vhc_pos_dict, ped_pos_dict,
                         sensor_max_range):
    obj_visibility_dict = {}
    for obj_id in vhc_pos_dict:
        if obj_id != self_vhc_id:
            obj_visibility_dict[(obj_id, 'Car')] = VisibilityRatio(percent=0.0, total_angle=0.0)
    for obj_id in ped_pos_dict:
        obj_visibility_dict[(obj_id, 'Pedestrian')] = VisibilityRatio(percent=0.0, total_angle=0.0)
    cur_start_angle = None
    cur_start_dist = None
    cur_obj = None
    for (angle_ind, cur_angle) in enumerate(important_angles):
        (closest_line_ind, line_dist) = get_closest_line_on_a_direction(cur_angle, all_lines)
        if closest_line_ind is not None:
            closest_obj = (all_lines[closest_line_ind].object_id, all_lines[closest_line_ind].object_type)
            if cur_obj is None:
                cur_obj = closest_obj
                cur_start_angle = cur_angle
                cur_start_dist = line_dist
            elif cur_obj == closest_obj:
                if angle_ind == len(important_angles) - 1:
                    visible_angle = cur_angle - cur_start_angle
                    if (0.0 < line_dist < sensor_max_range and 0.0 < cur_start_dist < sensor_max_range and
                            visible_angle > 0.0):
                        obj_visibility_dict[cur_obj].total_angle += visible_angle
                        obj_total_angle = (obj_angles_dict[cur_obj][1] - obj_angles_dict[cur_obj][0])
                        obj_visibility_dict[cur_obj].percent = (
                                obj_visibility_dict[cur_obj].total_angle / obj_total_angle)
                        if obj_visibility_dict[cur_obj].percent > 1.01 or obj_visibility_dict[cur_obj].percent < 0.0:
                            print(
                                '------------------------------ ERROR IN PERCENT COMPUTATION -------------------------')
                            print('end of fov {} percent: {} total: {}'.format(cur_obj,
                                                                               obj_visibility_dict[cur_obj].percent,
                                                                               obj_total_angle * 180.0 / math.pi))
                        obj_visibility_dict[cur_obj].percent = min(1.0, max(0.0, obj_visibility_dict[cur_obj].percent))
                    cur_obj = None
                else:
                    next_angle = (cur_angle + important_angles[angle_ind+1]) / 2.0
                    (closest_line_ind_next, line_dist_next) = \
                        get_closest_line_on_a_direction(next_angle, all_lines)
                    if closest_line_ind_next is not None:
                        closest_obj_next = (all_lines[closest_line_ind_next].object_id,
                                            all_lines[closest_line_ind_next].object_type)
                        if closest_obj_next != cur_obj:
                            visible_angle = cur_angle - cur_start_angle
                            if (0.0 < line_dist < sensor_max_range and 0.0 < cur_start_dist < sensor_max_range and
                                    visible_angle > 0.0):
                                obj_visibility_dict[cur_obj].total_angle += visible_angle
                                obj_total_angle = (obj_angles_dict[cur_obj][1] - obj_angles_dict[cur_obj][0])
                                obj_visibility_dict[cur_obj].percent = (
                                            obj_visibility_dict[cur_obj].total_angle / obj_total_angle)
                                if (obj_visibility_dict[cur_obj].percent > 1.01 or
                                        obj_visibility_dict[cur_obj].percent < 0.0):
                                    print('------------------ ERROR IN PERCENT COMPUTATION -------------------------')
                                    print('end switch {} percent: {} total: {}'.format(
                                        cur_obj, obj_visibility_dict[cur_obj].percent,
                                        obj_total_angle * 180.0 / math.pi))
                                obj_visibility_dict[cur_obj].percent = min(1.0, max(0.0, obj_visibility_dict[
                                    cur_obj].percent))
                            if closest_obj_next is not None:
                                cur_obj = closest_obj
                                cur_start_angle = cur_angle
                                cur_start_dist = line_dist
                    else:
                        visible_angle = cur_angle - cur_start_angle
                        if (0.0 < line_dist < sensor_max_range and 0.0 < cur_start_dist < sensor_max_range and
                                visible_angle > 0.0):
                            obj_visibility_dict[cur_obj].total_angle += visible_angle
                            obj_total_angle = (obj_angles_dict[cur_obj][1] - obj_angles_dict[cur_obj][0])
                            obj_visibility_dict[cur_obj].percent = (
                                    obj_visibility_dict[cur_obj].total_angle / obj_total_angle)
                            if obj_visibility_dict[cur_obj].percent > 1.01 or obj_visibility_dict[cur_obj].percent < 0:
                                print(
                                    '---------------------- ERROR IN PERCENT COMPUTATION -------------------------')
                                print('end of obj {} percent: {} total: {}'.format(
                                    cur_obj, obj_visibility_dict[cur_obj].percent, obj_total_angle * 180.0 / math.pi))
                            obj_visibility_dict[cur_obj].percent = min(1.0,
                                                                       max(0.0, obj_visibility_dict[cur_obj].percent))
                        cur_obj = None
            elif cur_obj != closest_obj:
                prev_angle = (cur_angle + important_angles[angle_ind - 1]) / 2.0
                (closest_line_ind_prev, line_dist_prev) = \
                    get_closest_line_on_a_direction(prev_angle, all_lines)
                if closest_line_ind_prev is not None:
                    closest_obj_prev = (all_lines[closest_line_ind_prev].object_id,
                                        all_lines[closest_line_ind_prev].object_type)
                    if closest_obj_prev == cur_obj:
                        if cur_obj in obj_visibility_dict:
                            visible_angle = cur_angle - cur_start_angle
                            if (0.0 < line_dist < sensor_max_range and 0.0 < cur_start_dist < sensor_max_range and
                                    visible_angle > 0.0):
                                obj_visibility_dict[cur_obj].total_angle += visible_angle
                                obj_total_angle = (obj_angles_dict[cur_obj][1] - obj_angles_dict[cur_obj][0])
                                obj_visibility_dict[cur_obj].percent = (obj_visibility_dict[cur_obj].total_angle /
                                                                        obj_total_angle)
                                if (obj_visibility_dict[cur_obj].percent > 1.01 or
                                        obj_visibility_dict[cur_obj].percent < 0.0):
                                    print('------------- ERROR IN PERCENT COMPUTATION -------------------------')
                                    print('end obstruct {} percent: {} total: {}'.format(
                                        cur_obj, obj_visibility_dict[cur_obj].percent,
                                        obj_total_angle * 180.0 / math.pi))
                                obj_visibility_dict[cur_obj].percent = min(1.0, max(0.0, obj_visibility_dict[
                                    cur_obj].percent))
                cur_obj = closest_obj
                cur_start_angle = cur_angle
                cur_start_dist = line_dist
        else:
            cur_obj = None
    return obj_visibility_dict


class VisibilityEvaluator(object):
    """VisibilityController class handles the generation of detection boxes and other ground truth
    information from the given camera, object position and rotation information."""
    def __init__(self):
        self.sensors = []

    def add_sensor(self, sensor):
        self.sensors.append(sensor)
        return len(self.sensors) - 1

    def set_sensor_parameters(self, sensor_ind, sensor_name=None, horizontal_fov=None, vertical_fov=None,
                              max_range=None, local_position=None, local_rotation=None, x_rotation=None):
        """Set sensor parameters."""
        if self.sensors and len(self.sensors) > sensor_ind:
            if horizontal_fov is not None:
                self.sensors[sensor_ind].hor_fov = horizontal_fov
            if vertical_fov is not None:
                self.sensors[sensor_ind].ver_fov = vertical_fov
            if max_range is not None:
                self.sensors[sensor_ind].max_range = max_range
            if local_position is not None:
                self.sensors[sensor_ind].local_position = np.array(local_position)
                self.sensors[sensor_ind].local_position.shape = (3, 1)
            if local_rotation is not None:
                self.sensors[sensor_ind].local_rotation = local_rotation
            if x_rotation is not None:
                self.sensors[sensor_ind].set_rotation_x_axis(x_rotation)
            if sensor_name is not None:
                self.sensors[sensor_ind].name = sensor_name

    def compute_sensor_position_rotation(self, sensor_ind, vhc_position, vhc_rotation):
        """Computes sensor position and rotation in world coordinates by using current vehicle position and rotation"""
        sensor_position = \
            convert_from_local_to_world_coordinates(vhc_rotation, vhc_position, self.sensors[sensor_ind].local_position)
        sensor_rot = np.array(vhc_rotation)
        sensor_rot.shape = (3, 3)
        sensor_rotation = np.matmul(self.sensors[sensor_ind].local_rotation, sensor_rot)
        return sensor_position, sensor_rotation

    def get_all_obj_lines_and_angles(self,
                                     self_vhc_id,
                                     sensor_ind,
                                     vhc_pos_dict,
                                     vhc_rot_dict,
                                     vhc_corners_dict,
                                     ped_pos_dict,
                                     ped_rot_dict,
                                     ped_corners_dict):
        all_lines = []
        obj_angles_dict = {}

        if self_vhc_id in vhc_pos_dict:
            vhc_position = np.array(vhc_pos_dict[self_vhc_id])
            vhc_position.shape = (3, 1)
            vhc_rotation = np.array(vhc_rot_dict[self_vhc_id])
            vhc_rotation.shape = (3, 3)
            (sensor_position, sensor_rotation) = \
                self.compute_sensor_position_rotation(sensor_ind, vhc_position, vhc_rotation)
            for obj_id in ped_pos_dict:
                if obj_id in ped_rot_dict and obj_id in ped_corners_dict:  # Defensive check
                    obj_class_name = 'Pedestrian'
                    (lines, left_angle, right_angle) = get_object_lines_and_angles(ped_pos_dict[obj_id],
                                                                                   ped_rot_dict[obj_id],
                                                                                   ped_corners_dict[obj_id],
                                                                                   sensor_position,
                                                                                   sensor_rotation,
                                                                                   obj_id,
                                                                                   obj_class_name)
                    for line in lines:
                        if not (line.right_angle < -self.sensors[sensor_ind].hor_fov and
                                line.left_angle > self.sensors[sensor_ind].hor_fov):
                            all_lines.append(line)
                    obj_angles_dict[(obj_id, obj_class_name)] = (left_angle, right_angle)
            for obj_id in vhc_pos_dict:
                if obj_id != self_vhc_id and (obj_id in vhc_rot_dict and obj_id in vhc_corners_dict):  # Defensive check
                    obj_class_name = 'Car'
                    (lines, left_angle, right_angle) = get_object_lines_and_angles(vhc_pos_dict[obj_id],
                                                                                   vhc_rot_dict[obj_id],
                                                                                   vhc_corners_dict[obj_id],
                                                                                   sensor_position,
                                                                                   sensor_rotation,
                                                                                   obj_id,
                                                                                   obj_class_name)
                    for line in lines:
                        if not (line.right_angle < -self.sensors[sensor_ind].hor_fov and
                                line.left_angle > self.sensors[sensor_ind].hor_fov):
                            all_lines.append(line)
                    obj_angles_dict[(obj_id, obj_class_name)] = (left_angle, right_angle)
        return all_lines, obj_angles_dict

    def get_all_vhc_and_ped_visibility_info(self,
                                            self_vhc_id,
                                            sensor_ind,
                                            vhc_pos_dict,
                                            vhc_rot_dict,
                                            vhc_corners_dict,
                                            ped_pos_dict,
                                            ped_rot_dict,
                                            ped_corners_dict):
        (all_lines, obj_angles_dict) = self.get_all_obj_lines_and_angles(self_vhc_id,
                                                                         sensor_ind,
                                                                         vhc_pos_dict,
                                                                         vhc_rot_dict,
                                                                         vhc_corners_dict,
                                                                         ped_pos_dict,
                                                                         ped_rot_dict,
                                                                         ped_corners_dict)
        important_angles = get_important_angles_from_lines(all_lines,
                                                           -self.sensors[sensor_ind].hor_fov,
                                                           self.sensors[sensor_ind].hor_fov)
        obj_visibility_dict = \
            compute_visibilities(important_angles, all_lines, obj_angles_dict, self_vhc_id, vhc_pos_dict, ped_pos_dict,
                                 self.sensors[sensor_ind].max_range)

        return obj_visibility_dict
