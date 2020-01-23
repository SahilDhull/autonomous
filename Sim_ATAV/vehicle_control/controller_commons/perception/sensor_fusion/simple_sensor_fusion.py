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
from Sim_ATAV.vehicle_control.controller_commons.perception.sensing.object_detection import ObjectDetection
from Sim_ATAV.vehicle_control.controller_commons import sensor_detection_tools
from Sim_ATAV.vehicle_control.controller_commons import controller_commons
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation.sensor_object import SensorObject
from Sim_ATAV.vehicle_control.controller_commons.perception.sensor_fusion.sensor_fusion_tracker \
    import SensorFusionTracker
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation import camera_to_object


class SimpleSensorFusion(object):
    def __init__(self, ego_state):
        self.object_detector = ObjectDetection()
        self.config_lidar_radar_distance_match_threshold = 8.0
        self.config_old_new_distance_match_threshold = 8.0
        self.config_camera_distance_match_threshold = 16.0

        self.config_front_triangle_line1_m_car = -192 / 126  # old value: -0.6  # Line 1 m for front triangle.
        self.config_front_triangle_line1_b_car = 1142.9  # Old value: 526  # Line 1 b for front triangle.
        self.config_front_triangle_line2_m_car = 192 / 126  # old value: 0.6  # Line 2 m for front triangle.
        self.config_front_triangle_line2_b_car = -758.9  # Old value: -202  # Line 2 b for front triangle.

        self.config_front_triangle_line1_m_ped = -192 / 204  # old value: -0.6  # Line 1 m for front triangle.
        self.config_front_triangle_line1_b_ped = 779.3  # Old value: 526  # Line 1 b for front triangle.
        self.config_front_triangle_line2_m_ped = 192 / 204  # old value: 0.6  # Line 2 m for front triangle.
        self.config_front_triangle_line2_b_ped = -395.3  # Old value: -202  # Line 2 b for front triangle.
        self.new_detections = []
        self.projected_old_objects = []
        self.ego_state = ego_state

    def get_detections(self):
        return self.new_detections

    def register_sensor(self, sensor_detector, sensor_period):
        """Register a new sensor."""
        self.object_detector.register_sensor(sensor_detector=sensor_detector, sensor_period=sensor_period)

    def update_detections(self, cur_time_ms):
        """Detect and update the tracked objects."""
        self.project_old_objects_to_now(cur_time_ms)
        self.detect_objects(cur_time_ms)
        self.merge_old_new_detections(cur_time_ms)

    def detect_objects(self, cur_time_ms):
        """Detect objects using sensors."""
        self.object_detector.detect_objects(cur_time_ms=cur_time_ms)
        self.merge_sensor_detections(cur_time_ms=cur_time_ms)

    def merge_sensor_detections(self, cur_time_ms):
        """Merge new detections from different sensors"""
        self.new_detections = []

        # First fill the list with lidar objects.
        if self.object_detector.is_lidar_read:
            for lidar_object in self.object_detector.lidar_objects:
                self.new_detections.append(lidar_object)
                self.new_detections[-1].object_speed_m_s = self.ego_state.get_speed_ms()
                self.new_detections[-1].object_direction = [0, -1]
                self.new_detections[-1].update_time = cur_time_ms

        # Then, try to match RADAR objects with the list.
        # Update the matching ones, insert unmatched radar objects to the list
        if self.object_detector.is_radar_read:
            [matches, unmatched_radar, _unmatched_existing] = \
                sensor_detection_tools.match_objects_in_sets(self.object_detector.radar_objects,
                                                             self.new_detections,
                                                             self.config_lidar_radar_distance_match_threshold)
            for match_ind in range(matches.shape[0]):
                match = matches[match_ind, :]
                new_ind = match[1]
                radar_ind = match[0]
                # Update speed information with radar measurement:
                self.new_detections[new_ind].object_position[0] = \
                    (self.new_detections[new_ind].object_position[0] +
                     self.object_detector.radar_objects[radar_ind].object_position[0]) / 2.0
                self.new_detections[new_ind].object_position[1] = \
                    (self.new_detections[new_ind].object_position[1] +
                     self.object_detector.radar_objects[radar_ind].object_position[1]) / 2.0
                self.new_detections[new_ind].sensor_recorded_position = self.new_detections[new_ind].object_position[:]
                self.new_detections[new_ind].object_speed_m_s = \
                    self.object_detector.radar_objects[radar_ind].object_speed_m_s
                self.new_detections[new_ind].set_aux_sensor_data(
                    SensorObject.SENSOR_RADAR,
                    self.object_detector.radar_objects[radar_ind].sensor_aux_data_dict[SensorObject.SENSOR_RADAR])
            for obj_ind in unmatched_radar:
                self.new_detections.append(self.object_detector.radar_objects[obj_ind])
                self.new_detections[-1].update_time = cur_time_ms

        # Finally, try to match camera objects with the list.
        # Update the matching ones, but DO NOT insert unmatched camera objects to the list
        # (unless they are at a very risky area)
        if self.object_detector.is_camera_read:
            [matches, unmatched_camera, _unmatched_existing] = \
                sensor_detection_tools.match_objects_in_sets(self.object_detector.camera_objects,
                                                             self.new_detections,
                                                             self.config_camera_distance_match_threshold)
            for match_ind in range(matches.shape[0]):
                match = matches[match_ind, :]
                new_ind = match[1]
                camera_ind = match[0]
                # Update object class information with camera detection:
                self.new_detections[new_ind].object_type = self.object_detector.camera_objects[camera_ind].object_type
                self.new_detections[new_ind].set_aux_sensor_data(
                    SensorObject.SENSOR_CAMERA,
                    self.object_detector.camera_objects[camera_ind].sensor_aux_data_dict[SensorObject.SENSOR_CAMERA])
        else:
            unmatched_camera = []

        if not (self.object_detector.has_lidar or self.object_detector.has_radar):
            # We only have camera in this case. We have to rely on camera and add all objects from camera.
            for camera_ind in unmatched_camera:
                self.new_detections.append(self.object_detector.camera_objects[camera_ind])
                self.new_detections[-1].update_time = cur_time_ms
        elif self.object_detector.has_lidar:
            # We don't have radar but have lidar.
            if self.object_detector.is_lidar_read:
                # Lidar is also read and camera detected an object which the lidar could not.
                for camera_ind in unmatched_camera:
                    if 0.0 < self.object_detector.camera_objects[camera_ind].object_position[1] < 40:
                        camera_det_obj = self.object_detector.camera_objects[camera_ind].sensor_aux_data_dict[
                            SensorObject.SENSOR_CAMERA]
                        det_class = self.object_detector.camera_objects[camera_ind].object_type
                        det_x_pos = camera_det_obj.detection_box[0]
                        det_y_pos = camera_det_obj.detection_box[1]
                        det_width = camera_det_obj.detection_box[2]
                        det_height = camera_det_obj.detection_box[3]
                        is_in_scope = (self.is_in_scope_triangle(det_x_pos + det_width / 2.0,
                                                                 min(383, det_y_pos + det_height / 2), det_class)
                                       or self.is_in_scope_triangle(det_x_pos - det_width / 2.0,
                                                                    min(383, det_y_pos + det_height / 2), det_class)
                                       or self.is_in_scope_triangle(det_x_pos, min(383, det_y_pos + det_height / 2),
                                                                    det_class))
                        if is_in_scope:
                            self.new_detections.append(self.object_detector.camera_objects[camera_ind])
                            # debug_print('Added camera object not detected by lidar: at {}'.format(
                            #    camera_detected_objects[camera_ind].object_position))
        for det_ind in range(len(self.new_detections)):
            self.new_detections[det_ind].global_position = controller_commons.convert_relative_to_global_position(
                object_relative_position=self.new_detections[det_ind].object_position,
                ego_global_position=self.ego_state.get_position(), ego_global_yaw_angle=self.ego_state.get_yaw_angle())
            self.new_detections[det_ind].sensor_recorded_global_position = \
                self.new_detections[det_ind].global_position[:]

    def project_old_objects_to_now(self, cur_time_ms):
        # Following is the list of old detections, projected to current time by object tracker (Kalman Filter)
        self.projected_old_objects = []
        for old_det_object in self.new_detections:
            projected_object = old_det_object
            # debug_print('sensor recorded direction in old obj: {}'.format(old_det_object.sensor_recorded_direction))
            projected_object.is_first_detection = False
            # debug_print('Before projection: {}'.format(old_det_object.object_position))
            projected_object.history.append((old_det_object.object_position[:],
                                             old_det_object.object_speed_m_s,
                                             old_det_object.object_direction[:],
                                             old_det_object.update_time,
                                             old_det_object.object_yaw_angle,
                                             old_det_object.object_yaw_rate))
            if old_det_object.tracker is not None:
                # debug_print('old obj history appended: {}'.format(projected_object.history[-1]))
                projected_object.tracker.tracked_object_state = \
                    projected_object.tracker.get_projection_with_state(
                        projected_object.tracker.tracked_object_state,
                        time_step=0.1,
                        time_duration=(cur_time_ms - projected_object.update_time) / 1000.0)
                obj_relative_states = \
                    self.convert_global_states_to_relative(projected_object.tracker.tracked_object_state)
                projected_object.object_position = obj_relative_states[0:2]
                projected_object.object_speed_m_s = obj_relative_states[2]
                projected_object.object_direction = [-math.sin(obj_relative_states[3]),
                                                     math.cos(obj_relative_states[3])]
                projected_object.object_yaw_angle = obj_relative_states[3]
                projected_object.object_yaw_rate = obj_relative_states[4]
                projected_object.global_position = projected_object.tracker.tracked_object_state[0:2]
                projected_object.global_speed = projected_object.tracker.tracked_object_state[2]
                projected_object.update_time = cur_time_ms
                # debug_print('After projection: {}'.format(projected_object.object_position))
                # debug_print('projected_object.tracker.tracked_object_state: {}'.format(
                #    projected_object.tracker.tracked_object_state))
            self.projected_old_objects.append(projected_object)

    def merge_old_new_detections(self, cur_time_ms):
        # Associate new and old detections
        [matches, _unmatched_new, unmatched_old] = \
            sensor_detection_tools.match_objects_in_sets(self.new_detections,
                                                         self.projected_old_objects,
                                                         distance_threshold=self.ego_state.get_speed_ms())

        for match_ind in range(matches.shape[0]):
            match = matches[match_ind, :]
            new_ind = match[0]
            old_ind = match[1]
            self.new_detections[new_ind].is_first_detection = False
            if self.projected_old_objects[old_ind].sensor_recorded_global_position is not None:
                # Compute global yaw angle, global speed and global yaw rate for the object.
                old_time = self.projected_old_objects[old_ind].detection_time
                global_motion_vector = [self.new_detections[new_ind].global_position[0] -
                                        self.projected_old_objects[old_ind].sensor_recorded_global_position[0],
                                        self.new_detections[new_ind].global_position[1] -
                                        self.projected_old_objects[old_ind].sensor_recorded_global_position[1]]
                self.new_detections[new_ind].global_yaw_angle = math.atan2(-global_motion_vector[0],
                                                                           global_motion_vector[1])
                global_motion_length = np.linalg.norm(np.array(global_motion_vector))
                self.new_detections[new_ind].global_speed = (global_motion_length * 1000.0) / (cur_time_ms - old_time)
                if self.projected_old_objects[old_ind].global_yaw_angle is not None:
                    yaw_rate = (((self.new_detections[new_ind].global_yaw_angle -
                                  self.projected_old_objects[old_ind].global_yaw_angle) * 1000.0) /
                                (cur_time_ms - old_time))
                    if abs(yaw_rate) < math.pi:
                        # if object is turning very fast,
                        # discard the computed yaw_rate because it is probably incorrect.
                        self.new_detections[new_ind].global_yaw_rate = yaw_rate

                old_pos = self.projected_old_objects[old_ind].sensor_recorded_position
                new_pos = self.new_detections[new_ind].object_position
                motion_vector = [new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]]
                motion_length = np.linalg.norm(np.array(motion_vector))
                if motion_length > 0.0001:
                    self.new_detections[new_ind].object_direction = [motion_vector[0] / motion_length,
                                                                     motion_vector[1] / motion_length]
                    self.new_detections[new_ind].sensor_recorded_direction = \
                        self.new_detections[new_ind].object_direction[:]
                    self.new_detections[new_ind].object_speed_m_s = (motion_length / (cur_time_ms - old_time)) * 1000.0

                if self.new_detections[new_ind].object_type == SensorObject.OBJECT_PEDESTRIAN:
                    obj_type_for_tracker = 'pedestrian'
                else:
                    obj_type_for_tracker = 'car'
                if self.object_detector.has_lidar or self.object_detector.has_radar:
                    # If it is only camera, tracking is completely useless and gives wrong information
                    obj_global_states = self.get_obj_global_states(self.new_detections[new_ind])
                    self.new_detections[new_ind].tracker = SensorFusionTracker(initial_state_mean=obj_global_states,
                                                                               object_type=obj_type_for_tracker)
                    # print('local: {} global: {}'.format(self.new_detections[new_ind].object_position,
                    #                                     obj_global_states))
                    # print('self: {}, {}, {}, {}'.format(self.ego_state.get_position(), self.ego_state.get_yaw_angle(),
                    #                                     self.ego_state.get_speed_ms(), self.ego_state.yaw_rate))
                    # sys.stdout.flush()

            for history_record in self.projected_old_objects[old_ind].history:
                self.new_detections[new_ind].history.append(history_record)
            # We use camera object detection less frequently.
            # If old detection of this object has a camera classification, use it.
            if (SensorObject.SENSOR_CAMERA not in self.new_detections[new_ind].sensor_aux_data_dict
                    and SensorObject.SENSOR_CAMERA in self.projected_old_objects[old_ind].sensor_aux_data_dict):
                self.new_detections[new_ind].object_type = camera_to_object.camera_to_sensor_class(
                    self.projected_old_objects[old_ind].sensor_aux_data_dict[SensorObject.SENSOR_CAMERA].object_class)
                self.new_detections[new_ind].sensor_aux_data_dict[SensorObject.SENSOR_CAMERA] = \
                    self.projected_old_objects[old_ind].sensor_aux_data_dict[SensorObject.SENSOR_CAMERA]
                if self.new_detections[new_ind].tracker is not None:
                    if self.new_detections[new_ind].object_type == SensorObject.OBJECT_PEDESTRIAN:
                        self.new_detections[new_ind].tracker.set_object_type('pedestrian')
                    else:
                        self.new_detections[new_ind].tracker.set_object_type('car')

        # Do not forget old detections for 1 sec if they are in the region of interest.
        for unmatched_old_ind in unmatched_old:
            if self.projected_old_objects[unmatched_old_ind].detection_time > cur_time_ms - 1000.0:
                vector_to_obj = np.array(self.projected_old_objects[unmatched_old_ind].object_position)
                if (np.linalg.norm(vector_to_obj) < 100.0 and
                    abs(math.atan2(vector_to_obj[0], vector_to_obj[1])) < math.pi / 8.0) \
                        or np.linalg.norm(vector_to_obj) < 50.0 and \
                        abs(math.atan2(vector_to_obj[0], vector_to_obj[1])) < math.pi / 4.0:
                    self.new_detections.append(self.projected_old_objects[unmatched_old_ind])
                    if self.projected_old_objects[unmatched_old_ind].detection_time < cur_time_ms - 501.0:
                        self.new_detections[-1].is_old_object = True

    def is_in_scope_triangle(self, pixel_x, pixel_y, obj_class):
        """Check if object is in front of the car."""
        if obj_class == SensorObject.OBJECT_PEDESTRIAN:
            in_triangle = \
                (pixel_y > self.config_front_triangle_line1_m_ped*pixel_x + self.config_front_triangle_line1_b_ped and
                 pixel_y > self.config_front_triangle_line2_m_ped*pixel_x + self.config_front_triangle_line2_b_ped)
        else:
            in_triangle = \
                (pixel_y > self.config_front_triangle_line1_m_car*pixel_x + self.config_front_triangle_line1_b_car and
                 pixel_y > self.config_front_triangle_line2_m_car*pixel_x + self.config_front_triangle_line2_b_car)
        return in_triangle

    def get_obj_global_states(self, obj):
        """Converts object states from local(relative) to global(absolute) states."""
        obj_global_pos = obj.global_position if obj.global_position is not None else \
            controller_commons.convert_relative_to_global_position(object_relative_position=obj.object_position,
                                                                   ego_global_position=self.ego_state.get_position(),
                                                                   ego_global_yaw_angle=self.ego_state.get_yaw_angle())
        if obj.global_yaw_angle is not None:
            obj_global_yaw = obj.global_yaw_angle
        else:
            obj_global_yaw = self.ego_state.get_yaw_angle() + math.pi
            if obj_global_yaw > 2.0 * math.pi:
                obj_global_yaw -= 2.0 * math.pi
        obj_global_yaw_rate = obj.global_yaw_rate if obj.global_yaw_rate is not None else 0.0
        obj_global_speed = obj.global_speed if obj.global_speed is not None else 0.0
        return [obj_global_pos[0], obj_global_pos[1], obj_global_speed, obj_global_yaw, obj_global_yaw_rate]

    def convert_global_states_to_relative(self, obj_global_state):
        obj_relative_position = controller_commons.convert_global_to_relative_position(
            object_global_position=obj_global_state[0:2], ego_global_position=self.ego_state.get_position(),
            ego_global_yaw_angle=self.ego_state.get_yaw_angle())
        obj_relative_yaw = obj_global_state[3] - self.ego_state.get_yaw_angle()
        obj_relative_yaw_rate = obj_global_state[4] - self.ego_state.yaw_rate
        obj_global_speed_x = -obj_global_state[2] * math.sin(obj_global_state[3])
        obj_global_speed_y = obj_global_state[2] * math.cos(obj_global_state[3])
        ego_speed_x = -self.ego_state.get_speed_ms() * math.sin(self.ego_state.get_yaw_angle())
        ego_speed_y = self.ego_state.get_speed_ms() * math.cos(self.ego_state.get_yaw_angle())
        obj_relative_speed_x = obj_global_speed_x - ego_speed_x
        obj_relative_speed_y = obj_global_speed_y - ego_speed_y
        obj_relative_speed = math.sqrt(obj_relative_speed_x ** 2 + obj_relative_speed_y ** 2)
        return [obj_relative_position[0], obj_relative_position[1], obj_relative_speed, obj_relative_yaw, obj_relative_yaw_rate]
