"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import math
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation.sensor_object import SensorObject
from Sim_ATAV.vehicle_control.controller_commons import controller_commons


class RadioDetection(object):
    def __init__(self, controller_communication_interface, ego_vhc_id):
        self.contr_comm = controller_communication_interface
        self.ego_vhc_id = ego_vhc_id
        self.vhc_pos_dict = {}
        self.ped_pos_dict = {}
        self.vhc_rot_dict = {}
        self.last_record_time = 0
        self.detected_objects = []
        self.detection_index_dict = {}
        self.vhc_corners_dict = {}
        self.ped_corners_dict = {}
        self.ped_rot_dict = {}

    def update_detections(self, command_list, cur_time_ms):
        """Read sensor-like information from Simulation Supervisor."""
        self.detected_objects = []
        prev_vhc_pos_dict = self.vhc_pos_dict
        prev_ped_pos_dict = self.ped_pos_dict
        self.vhc_pos_dict = self.contr_comm.get_all_vehicle_positions(command_list)
        self.ped_pos_dict = self.contr_comm.get_all_pedestrian_positions(command_list)
        self.vhc_rot_dict = self.contr_comm.get_all_vehicle_rotations(command_list)
        vhc_corners_dict_temp = self.contr_comm.get_all_vehicle_box_corners(command_list)
        if len(vhc_corners_dict_temp) > 0:
            self.vhc_corners_dict = vhc_corners_dict_temp.copy()
        ped_corners_dict_temp = self.contr_comm.get_all_pedestrian_box_corners(command_list)
        if len(ped_corners_dict_temp) > 0:
            self.ped_corners_dict = ped_corners_dict_temp.copy()
        self.ped_rot_dict = self.contr_comm.get_all_pedestrian_rotations(command_list)
        self.detection_index_dict = {}

        if self.ego_vhc_id in self.vhc_pos_dict:
            received_self_pos = self.vhc_pos_dict[self.ego_vhc_id]
            received_self_rot = self.vhc_rot_dict[self.ego_vhc_id]
            received_self_orientation = controller_commons.rotation_matrix_to_azimuth(received_self_rot)
        else:
            received_self_pos = [0, 0, 0]
            received_self_orientation = 0.0
        if self.ego_vhc_id in prev_vhc_pos_dict:
            received_self_prev_pos = prev_vhc_pos_dict[self.ego_vhc_id]
            received_self_prev_rot = self.vhc_rot_dict[self.ego_vhc_id]
            received_self_prev_orientation = controller_commons.rotation_matrix_to_azimuth(received_self_prev_rot)
        else:
            received_self_prev_pos = received_self_pos[:]
            received_self_prev_orientation = 0.0

        # Add Vehicles
        for received_obj_id in self.vhc_pos_dict:
            if received_obj_id != self.ego_vhc_id:
                det_object = self.reception_to_sensor_detection_object(received_obj_id,
                                                                       self.vhc_pos_dict,
                                                                       prev_vhc_pos_dict,
                                                                       received_self_pos,
                                                                       received_self_prev_pos,
                                                                       received_self_orientation,
                                                                       received_self_prev_orientation,
                                                                       SensorObject.OBJECT_CAR,
                                                                       cur_time_ms)
                det_object.set_object_seen_by_sensor('receiver')
                self.detected_objects.append(det_object)
                self.detection_index_dict[('vehicle', received_obj_id)] = len(self.detected_objects) - 1
        # Add Pedestrians
        for received_obj_id in self.ped_pos_dict:
            det_object = self.reception_to_sensor_detection_object(received_obj_id,
                                                                   self.ped_pos_dict,
                                                                   prev_ped_pos_dict,
                                                                   received_self_pos,
                                                                   received_self_prev_pos,
                                                                   received_self_orientation,
                                                                   received_self_prev_orientation,
                                                                   SensorObject.OBJECT_PEDESTRIAN,
                                                                   cur_time_ms)
            det_object.set_object_seen_by_sensor('receiver')
            self.detected_objects.append(det_object)
            self.detection_index_dict[('pedestrian', received_obj_id)] = len(self.detected_objects) - 1
        self.last_record_time = cur_time_ms
        return self.detected_objects

    def reception_to_sensor_detection_object(self,
                                             object_id,
                                             pos_dict,
                                             prev_pos_dict,
                                             self_position,
                                             prev_self_position,
                                             self_orientation,
                                             self_prev_orientation,
                                             object_type,
                                             cur_time_ms):
        """Creates a SensorObject from given info."""
        det_object = SensorObject(object_type)
        det_object.set_detection_time(cur_time_ms)
        # Received Object Position is in world coordinates
        # Compute distance and relative angle in order to compute relative x and y in accordance with self orientation.
        received_obj_pos = pos_dict[object_id]
        rel_x_world = received_obj_pos[0] - self_position[0]
        rel_y_world = received_obj_pos[2] - self_position[2]
        rel_d = math.sqrt(rel_x_world**2 + rel_y_world**2)
        rel_angle = -(math.atan2(rel_x_world, rel_y_world) + self_orientation)
        received_obj_rel_pos = controller_commons.polar_coordinates_to_cartesian(rel_d, rel_angle)
        det_object.set_object_position(received_obj_rel_pos)

        if object_id in prev_pos_dict:
            det_object.is_first_detection = False
            received_obj_prev_pos = prev_pos_dict[object_id]
            rel_x_world = received_obj_prev_pos[0] - prev_self_position[0]
            rel_y_world = received_obj_prev_pos[2] - prev_self_position[2]
            rel_d = math.sqrt(rel_x_world**2 + rel_y_world**2)
            rel_angle = -(math.atan2(rel_x_world, rel_y_world) + self_prev_orientation)
            received_obj_old_rel_pos = [rel_d * math.sin(rel_angle), rel_d * math.cos(rel_angle)]
            (relative_obj_direction, received_obj_speed, _d) = \
                controller_commons.position_to_direction_speed_distance(received_obj_rel_pos,
                                                                        received_obj_old_rel_pos,
                                                                        time_diff=(cur_time_ms -
                                                                                   self.last_record_time) / 1000.0)
        else:
            received_obj_speed = 0.0
            relative_obj_direction = [0, -1]
        det_object.set_object_speed_m_s(received_obj_speed)
        det_object.set_object_direction(relative_obj_direction)
        return det_object

    def get_detected_object_by_type(self, obj_type, obj_id):
        if (obj_type, obj_id) in self.detection_index_dict:
            return self.detected_objects[self.detection_index_dict[(obj_type, obj_id)]]
        else:
            return None
