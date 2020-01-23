"""Defines utility functions commonly used in different controllers
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
# import sys
import math
import struct
import numpy as np
from Sim_ATAV.common.coordinate_system import CoordinateSystem


def find_vehicles_in_lidar_data(lidar_points, lidar_device, jump_threshold=3.0):
    """Returns an array of start and end rays of lidar in which a vehicle is expected."""
    found_vehicles = []
    vhc_start_ind = 0
    vhc_start_x = 0
    vhc_start_z = 0
    lidar_range = lidar_device.getMaxRange()

    is_first = True
    num_points = len(lidar_points)
    prev_dist = 0
    for (pt_ind, lidar_point) in enumerate(lidar_points):
        dist = math.sqrt(lidar_point.x**2 + lidar_point.z**2)
        if is_first:
            vhc_start_x = lidar_point.x
            vhc_start_z = lidar_point.z
            vhc_start_ind = pt_ind
            is_first = False
        else:
            if dist < prev_dist - jump_threshold:
                # Update vhc_start no matter what.
                vhc_start_ind = pt_ind
                vhc_start_x = lidar_point.x
                vhc_start_z = lidar_point.z
            elif dist > prev_dist + jump_threshold:
                if (lidar_point.x - 7.0 < vhc_start_x < lidar_point.x + 7.0 and
                        lidar_point.z - 7.0 < vhc_start_z < lidar_point.z + 7.0):
                    if 1 < dist < lidar_range and abs(lidar_point.x - vhc_start_x) > 1.0:
                        found_vehicles.append((vhc_start_ind,
                                               pt_ind,
                                               [vhc_start_x, vhc_start_z],
                                               [lidar_point.x, lidar_point.z]))

                    vhc_start_ind = pt_ind
                    vhc_start_x = lidar_point.x
                    vhc_start_z = lidar_point.z
                else:
                    vhc_start_ind = pt_ind
                    vhc_start_x = lidar_point.x
                    vhc_start_z = lidar_point.z
            elif pt_ind == num_points - 1:
                # Last laser ray. There may be a vehicle on the right most region.
                if (lidar_point.x - 7.0 < vhc_start_x < lidar_point.x + 7.0 and
                        lidar_point.z - 7.0 < vhc_start_z < lidar_point.z + 7.0):
                    if 1 < dist < lidar_range:
                        found_vehicles.append((vhc_start_ind,
                                               pt_ind,
                                               [vhc_start_x, vhc_start_z],
                                               [lidar_point.x, lidar_point.z]))

        prev_dist = dist
    return found_vehicles


def compute_ang_wrt_pos_lat_axis(vector):
    """vector_ang is the angle of the vector wrt positive lat_axis"""
    if -0.00001 < vector[CoordinateSystem.LAT_AXIS] < 0.00001:
        if vector[CoordinateSystem.LONG_AXIS] >= 0:
            vector_ang = math.pi / 2.0
        else:
            vector_ang = -math.pi / 2.0
    elif vector[CoordinateSystem.LAT_AXIS] > 0:
        vector_ang = math.atan(vector[CoordinateSystem.LONG_AXIS]
                               / vector[CoordinateSystem.LAT_AXIS])
    elif vector[CoordinateSystem.LONG_AXIS] < 0:  # vector[CoordinateSystem.LAT_AXIS] < 0
        vector_ang = -math.pi + math.atan(vector[CoordinateSystem.LONG_AXIS]
                                          / vector[CoordinateSystem.LAT_AXIS])
    else:  # vector[CoordinateSystem.LONG_AXIS] >= 0 and vector[CoordinateSystem.LAT_AXIS] < 0
        vector_ang = math.pi + math.atan(vector[CoordinateSystem.LONG_AXIS]
                                         / vector[CoordinateSystem.LAT_AXIS])
    return vector_ang


def angle_between_vectors(vector1, vector2):
    """ Returns the angle in radians between vectors 'vector1' and 'vector2'::
             angle_between((1, 0, 0), (0, 1, 0)): 1.5707963267948966
             angle_between((1, 0, 0), (1, 0, 0)): 0.0
             angle_between((1, 0, 0), (-1, 0, 0)): 3.141592653589793"""

    return math.atan2(vector2[0], vector2[2]) - math.atan2(vector1[0], vector1[2])


def compute_loom_rate(intruder_point, ego_point, intruder_velocity, ego_linear_velocity):
    """Computes the looming rate. Based on looming points approach to detect future collisions."""
    vect_pt2ego = np.subtract(intruder_point, ego_point)
    loom_rate = \
        ((np.cross(vect_pt2ego, ego_linear_velocity) + np.cross(vect_pt2ego, intruder_velocity))
         / (np.linalg.norm(vect_pt2ego) ** 2))
    return loom_rate


def check_collision_course(ego_vhc_pos,
                           ego_vhc_pts,
                           ego_vhc_ang_velocity,
                           ego_vhc_velocity,
                           intruder_pts,
                           intruder_velocity):
    """ Checks if two vehicles are on a collision course.
        From:
        "Vehicle Collision Probability Calculation for General Traffic Scenarios Under Uncertainty"
        J. Ward, G. Agamennoni, S. Worrall, E. Nebot"""
    is_coll_path = False

    for ego_pt in ego_vhc_pts:
        # v_i_lin is a numpy array of linear velocity of the loom point.
        v_i_lin = \
            ego_vhc_velocity + np.cross(ego_vhc_ang_velocity, np.subtract(ego_pt, ego_vhc_pos))

        # Find points of intruder at the minimum and maximum angle wrt ego vehicle:
        min_ang = np.inf
        max_ang = -np.inf
        min_ang_pt = None
        max_ang_pt = None
        for intruder_point in intruder_pts:
            vector_ego_to_intruder = np.subtract(intruder_point, ego_pt)
            angle_ego_to_intruder = angle_between_vectors(vector_ego_to_intruder, v_i_lin)
            if angle_ego_to_intruder < min_ang:
                min_ang_pt = intruder_point
                min_ang = angle_ego_to_intruder
            if angle_ego_to_intruder > max_ang:
                max_ang_pt = intruder_point
                max_ang = angle_ego_to_intruder

        # Compute the loom rates for min and max angle points:
        loom_rate_min_ang_pt = compute_loom_rate(min_ang_pt, ego_pt, intruder_velocity, v_i_lin)
        loom_rate_max_ang_pt = compute_loom_rate(max_ang_pt, ego_pt, intruder_velocity, v_i_lin)
        if loom_rate_min_ang_pt[1] <= 0.0 <= loom_rate_max_ang_pt[1]:
            is_coll_path = True
            break
    return is_coll_path


def check_collision_course_on_pixels(obj_prev_left, obj_prev_right, obj_new_left, obj_new_right):
    right_movement = obj_new_right - obj_prev_right
    left_movement = obj_new_left - obj_prev_left
    if right_movement >= 0 >= left_movement:
        is_coll_path = True
    else:
        is_coll_path = False
    return is_coll_path


def get_receiver_message(receiver_device):
    """Returns the received message by the receiver device"""
    is_received = False
    if receiver_device.getQueueLength() > 0:
        received_message = []
        # Receive message
        while receiver_device.getQueueLength() > 0:
            received_message += receiver_device.getData()
            receiver_device.nextPacket()
        received_message = ''.join(received_message)
        is_received = True
    else:
        received_message = ''
    return is_received, received_message


def receive_vhc_pos(received_message, target_vhc_id):
    """Extracts a vehicle position from the received message."""
    pos = [0.0, 0.0, 0.0]
    # Evaluate message
    if len(received_message) > 0:
        # cmd = struct.unpack('B', received_message[0:struct.calcsize('B')])[0]
        # TODO: Check cmd for different type of messages
        cur_index = struct.calcsize('B')
        num_vehicles = \
            struct.unpack('h', received_message[cur_index:cur_index + struct.calcsize('h')])[0]
        cur_index += struct.calcsize('h')
        for i in range(num_vehicles):
            (vehicle_id, pos[0], pos[1], pos[2]) = struct.unpack(
                "Bddd", received_message[cur_index:cur_index + struct.calcsize("Bddd")])
            if vehicle_id == target_vhc_id:
                break
            cur_index += struct.calcsize("Bddd")
    return pos


def receive_all_vhc_pos(received_message):
    """Extracts all vehicle positions from the received message."""
    ret_dict = {}
    # Evaluate message
    if len(received_message) > 0:
        # cmd = struct.unpack('B', received_message[0:struct.calcsize('B')])[0]
        # TODO: Check cmd for different type of messages
        cur_index = struct.calcsize('B')
        num_vehicles = \
            struct.unpack('h', received_message[cur_index:cur_index + struct.calcsize('h')])[0]
        cur_index += struct.calcsize('h')
        for i in range(num_vehicles):
            pos = [0.0, 0.0, 0.0]
            (vehicle_id, pos[0], pos[1], pos[2]) = struct.unpack(
                "Bddd", received_message[cur_index:cur_index + struct.calcsize("Bddd")])
            ret_dict[vehicle_id] = [pos[0], pos[1], pos[2]]
            cur_index += struct.calcsize("Bddd")
    return ret_dict


def receive_nn_weights(received_message):
    """Receives a list of neural network weights to use in a controller."""
    nn_weights = []
    if len(received_message) > 0:
        cmd = struct.unpack('B', received_message[0:struct.calcsize('B')])[0]
        # TODO: Check cmd for different type of messages
        if cmd == 1:
            (temp, length) = struct.unpack('Bh', received_message[0:struct.calcsize('Bh')])
            cur_index = struct.calcsize('Bh')
            nn_weights = list(struct.unpack('%sd' % length, received_message[cur_index:]))
    return nn_weights


def kmh_to_ms(kmh):
    """Converts km/h to m/s"""
    return kmh / 3.6


def speed_ms_to_kmh(speed_ms):
    """Converts to m/s to km/h"""
    return speed_ms * 3.6


def get_bearing(compass_device):
    """Return the vehicle's heading in radians.
    When thw world's +ve x is on the left and +ve y is on top, angle is 0.
    Angle increases clockwise."""
    if compass_device is not None:
        compass_data = compass_device.getValues()
        radians = math.atan2(compass_data[2], -compass_data[0])
        while radians > math.pi:
            radians -= 2*math.pi
        while radians < -math.pi:
            radians += 2*math.pi
    else:
        radians = 0.0
    return radians


def convert_steering_value_to_rad(steering_value):
    """Converts the steering value in range -1, +1 to steering angle in radians."""
    return steering_value * 0.91  # Yes, that simple! Assuming linear relation! Correct for Prius.


def polar_coordinates_to_cartesian(distance, relative_angle):
    """Computes relative x and y for give distance and angle.
    Can be used to convert radar measurements to relative position."""
    # Here +ve x is towards the left but the angle is +ve clockwise.
    return [-1 * distance * math.sin(relative_angle), distance * math.cos(relative_angle)]


def position_to_direction_speed_distance(new_position, old_position, time_diff=None):
    """Computes the motion direction, speed and replacement distance
    from new and old positions"""
    motion_vector = np.array(new_position) - np.array(old_position)
    distance = np.linalg.norm(motion_vector)
    direction = motion_vector / max(0.5, distance)
    if time_diff is None:
        speed = 0.0
    else:
        speed = distance / time_diff
    return list(direction), speed, distance


def normalize_radian(angle, min_radian, max_radian):
    """Normalizes angle between min and max radians."""
    while angle > max_radian:
        angle -= 2.0*math.pi
    while angle < min_radian:
        angle += 2.0*math.pi
    return angle


def rotation_matrix_to_azimuth(rotation_matrix):
    """Computes azimuth from rotation matrix."""
    azimuth = -math.atan2(rotation_matrix[2], rotation_matrix[8])
    return normalize_radian(azimuth, -math.pi, math.pi)


def rotate_point_ccw(point, rotation_angle):
    cos_theta = math.cos(rotation_angle)
    sin_theta = math.sin(rotation_angle)
    return np.dot(np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]), point)


def convert_global_to_relative_position(object_global_position, ego_global_position, ego_global_yaw_angle):
    """convert the global position of the object to relative position wrt the ego vehicle."""
    obj_temp_rel = [object_global_position[0] - ego_global_position[0],
                    object_global_position[1] - ego_global_position[1]]
    obj_pos_rotated = rotate_point_ccw(np.transpose(np.array(obj_temp_rel)), -ego_global_yaw_angle)
    return [obj_pos_rotated[0], obj_pos_rotated[1]]


def convert_relative_to_global_position(object_relative_position, ego_global_position, ego_global_yaw_angle):
    """Convert the relative position of the object to global using the ego global position and yaw angle."""
    obj_pos_rotated = rotate_point_ccw(np.transpose(np.array(object_relative_position)), ego_global_yaw_angle)
    return [obj_pos_rotated[0] + ego_global_position[0], obj_pos_rotated[1] + ego_global_position[1]]


def read_gps_sensor(gps_device):
    """Reads GPS sensor."""
    if gps_device is not None:
        sensor_gps_speed_m_s = gps_device.getSpeed()
        sensor_gps_position_m = gps_device.getValues()
    else:
        sensor_gps_speed_m_s = 0.0
        sensor_gps_position_m = [0.0, 0.0, 0.0]
    return sensor_gps_position_m, sensor_gps_speed_m_s


def read_compass_sensor(compass_device):
    """Reads Compass Sensor."""
    if compass_device is not None:
        sensor_compass_bearing_rad = get_bearing(compass_device)
    else:
        sensor_compass_bearing_rad = 0.0
    return sensor_compass_bearing_rad
