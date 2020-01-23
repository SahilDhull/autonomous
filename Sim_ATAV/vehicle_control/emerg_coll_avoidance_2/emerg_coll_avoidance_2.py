"""Defines EmergencyCollAvoidance2 class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import math
import timeit
import numpy as np
from Sim_ATAV.common.controller_communication_interface import ControllerCommunicationInterface
from Sim_ATAV.vehicle_control.base_controller.base_controller import BaseCarController
from Sim_ATAV.vehicle_control.controller_commons.controller_commons import get_bearing
from Sim_ATAV.vehicle_control.generic_stanley_controller.generic_stanley_controller import GenericStanleyController
from Sim_ATAV.classifier.classifier_interface.classifier import Classifier
from Sim_ATAV.classifier.classifier_interface.classification_client import ClassificationClient
from Sim_ATAV.classifier.classifier_interface.object_tracker import ObjectTracker
from Sim_ATAV.classifier.classifier_interface.ground_truth_generator import GroundTruthGenerator
from Sim_ATAV.classifier.classifier_interface.performance_computations import iou_performance_for_object


class EmergencyCollAvoidance2(BaseCarController):
    """EmergencyCollAvoidance2 is a car controller class for Webots.
    It is used for experimenting basic interface with an object detection framework.
    When it detects a pedestrian in front of the car, it applies emergency braking,
    otherwise drives the car with given target throttle."""

    CAMERA_TO_FRONT_DISTANCE = 2.6  # 2.6 m is the distance from camera to Prius front face
    PED_DISTANCE_SCALE_FACTOR = 2793  # Focal length * known height for pedestrian
    CAR_DISTANCE_SCALE_FACTOR = 2483  # Focal length * known height for car
    # Defining a triangle with lines y1 = m1*x + b1 and y2 = m2*x+b2:
    CAR_FRONT_TRIANGLE_LINE1_M = -192/126  # old value: -0.6  # Line 1 m for front triangle.
    CAR_FRONT_TRIANGLE_LINE1_B = 1142.9  # Old value: 526  # Line 1 b for front triangle.
    CAR_FRONT_TRIANGLE_LINE2_M = 192/126  #old value: 0.6  # Line 2 m for front triangle.
    CAR_FRONT_TRIANGLE_LINE2_B = -758.9  # Old value: -202  # Line 2 b for front triangle.

    PED_FRONT_TRIANGLE_LINE1_M = -192/204  # old value: -0.6  # Line 1 m for front triangle.
    PED_FRONT_TRIANGLE_LINE1_B = 779.3  # Old value: 526  # Line 1 b for front triangle.
    PED_FRONT_TRIANGLE_LINE2_M = 192/204  #old value: 0.6  # Line 2 m for front triangle.
    PED_FRONT_TRIANGLE_LINE2_B = -395.3  # Old value: -202  # Line 2 b for front triangle.

    CAMERA_LOCAL_COORDINATES = [0.0, 1.3, 1.1]
    CAMERA_X_ROT_ANGLE = -0.01
    CAMERA_LOCAL_ROTATION = np.array([[1.0, 0.0, 0.0], \
                                      [0.0, math.cos(CAMERA_X_ROT_ANGLE), -math.sin(CAMERA_X_ROT_ANGLE)], \
                                      [0.0, math.sin(CAMERA_X_ROT_ANGLE), math.cos(CAMERA_X_ROT_ANGLE)]])

    CAR_BOX_COLOR = 0xFF0000
    CYCLIST_BOX_COLOR = 0x00FF00
    PEDESTRIAN_BOX_COLOR = 0x0000FF

    BOX_COLOR_DICT = {Classifier.CAR_CLASS_LABEL:CAR_BOX_COLOR,
                      Classifier.PEDESTRIAN_CLASS_LABEL:PEDESTRIAN_BOX_COLOR,
                      Classifier.CYCLIST_CLASS_LABEL:CYCLIST_BOX_COLOR}

    CAR_BOX_TEXT = 'Car'
    CYCLIST_BOX_TEXT = 'Bike'
    PEDESTRIAN_BOX_TEXT = 'Ped'

    DEBUG_DISPLAY = True

    BOX_NAME_DICT = {Classifier.CAR_CLASS_LABEL:CAR_BOX_TEXT,
                     Classifier.PEDESTRIAN_CLASS_LABEL:PEDESTRIAN_BOX_TEXT,
                     Classifier.CYCLIST_CLASS_LABEL:CYCLIST_BOX_TEXT}
    CLASSIFIER_PERIOD_MS = 50
    MIN_EMERGENCY_BRAKE_DURATION_MS = 100.0
    MEASURE_EXEC_TIME = False
    LANE_WIDTH = 3.5
    MIN_STEERING_MANEUVER_MS = 2000.0
    EMERGENCY_STEERING_TTC = 1.0

    def __init__(self, controller_parameters):
        (car_model, target_throttle, is_online_classifier, target_lat_pos, self_vhc_id) = controller_parameters
        BaseCarController.__init__(self, car_model)
        self.camera_device_name = "camera"
        self.camera = None
        self.compass_device_name = "compass"
        self.compass = None
        self.display_device_name = 'display'
        self.display = None
        self.gps_device_name = "gps"
        self.gps = None
        self.receiver_device_name = 'receiver'
        self.receiver = None
        self.emitter_device_name = 'emitter'
        self.emitter = None
        self.target_throttle = float(target_throttle)
        self.is_online_classifier = is_online_classifier in ('True', 'true', 'yes', 'Yes')
        print('Is online Classifier? {}'.format(self.is_online_classifier))
        self.classifier = None
        self.classification_client = None
        self.obj_tracker = None
        self.ground_truth_generator = None
        self.contr_comm = ControllerCommunicationInterface()
        self.target_lat_pos = float(target_lat_pos)
        self.target_bearing = 0.0
        self.lateral_controller = GenericStanleyController()
        self.lateral_controller.k = 0.5
        self.lateral_controller.set_output_range(-0.8, 0.8)
        self.self_vhc_id = int(self_vhc_id)
        print("pedestrian_avoidance Initialized: {}, {}".format(car_model, self.target_throttle))

    def compute_distance(self, object_class, pixel_height):
        """Convert detected object pixel height to distance to the object."""
        if pixel_height < 1:
            pixel_height = 1
        if object_class == Classifier.PEDESTRIAN_CLASS_LABEL:
            dist = self.PED_DISTANCE_SCALE_FACTOR / pixel_height
        else:
            dist = self.CAR_DISTANCE_SCALE_FACTOR / pixel_height
        return dist

    def is_in_scope_triangle(self, pixel_x, pixel_y, obj_class):
        """Check if object is in front of the car."""
        in_triangle = False
        if obj_class == Classifier.PEDESTRIAN_CLASS_LABEL:
            if pixel_y > self.PED_FRONT_TRIANGLE_LINE1_M*pixel_x + self.PED_FRONT_TRIANGLE_LINE1_B and \
                pixel_y > self.PED_FRONT_TRIANGLE_LINE2_M*pixel_x + self.PED_FRONT_TRIANGLE_LINE2_B:
                in_triangle = True
        else:
            if pixel_y > self.CAR_FRONT_TRIANGLE_LINE1_M*pixel_x + self.CAR_FRONT_TRIANGLE_LINE1_B and \
                pixel_y > self.CAR_FRONT_TRIANGLE_LINE2_M*pixel_x + self.CAR_FRONT_TRIANGLE_LINE2_B:
                in_triangle = True
        return in_triangle

    def refresh_display_overlay(self):
        if self.display is not None:
            # Remove objects:
            self.display.setAlpha(0.0)
            self.display.fillRectangle(0, 0, self.display.getWidth(), self.display.getHeight())
            self.display.setAlpha(1.0)
            # Set pedestrian detection box empty:
            self.display.setColor(0x0000FF)
            self.display.fillRectangle(0, 10, self.display.getWidth(), 10)

    def show_object_on_display(self, obj_class, detection_box, obj_ind, with_text=True):
        if self.display is not None:
            self.display.setColor(self.BOX_COLOR_DICT[obj_class])
            # detection_box array: [x, y, w, h]
            # Detection box from SqueezeDet has x and y as the center of image.
            # Webots expects x and y as the top left corner of the image.
            self.display.drawRectangle(int(detection_box[0] - detection_box[2]/2.0), \
                                       int(detection_box[1] - detection_box[3]/2.0), \
                                       int(detection_box[2]), \
                                       int(detection_box[3]))
            if with_text:
                self.display.drawText(self.BOX_NAME_DICT[obj_class] + '_' + str(obj_ind), \
                                    int(detection_box[0] - detection_box[2]/2.0) + 1, \
                                    int(detection_box[1] - detection_box[3]/2.0) + 1)

    def mark_critical_object_on_display(self, detection_box, color):
        if self.display is not None:
            self.display.setColor(color)
            self.display.setAlpha(0.7)
            # detection_box array: [x, y, w, h]
            # Detection box from SqueezeDet has x and y as the center of image.
            # Webots expects x and y as the top left corner of the image.
            self.display.fillRectangle(int(detection_box[0] - detection_box[2]/2.0), \
                                       min(int(detection_box[1] - detection_box[3]/2.0) + 10, int(detection_box[1] + detection_box[3]/2.0) - 1), \
                                       int(detection_box[2]), \
                                       max(1, int(detection_box[3] - 10)))
            self.display.setAlpha(1.0)

    def show_pedestrian_details_on_display(self, ped_count, dist, is_pedestrian_in_scope, obj_ind):
        if self.display is not None:
            self.display.setColor(0xFFFFFF)
            self.display.drawText('Ped_{0} Dist: {1:.1f} Front?: {2}'.format(obj_ind, dist, is_pedestrian_in_scope), 200*(ped_count - 1) + 1, 11)

    def show_control_information_on_display(self, throttle, steering, is_emergency_brake=False):
        if self.display is not None:
            self.display.setColor(0x00FF00)
            self.display.fillRectangle(0, 0, self.display.getWidth(), 10)
            self.display.setColor(0x0000FF)
            self.display.drawText('Throttle: {:.1f} Steering: {:.2f}'.format(throttle, steering), 1, 1)
            if is_emergency_brake:
                rect_start_coordinate = int(self.display.getWidth() / 2.0)
                self.display.setColor(0xFF0000)
                self.display.fillRectangle(rect_start_coordinate, 0, self.display.getWidth()-rect_start_coordinate, 10)
                self.display.setColor(0xFFFFFF)
                self.display.drawText('EMERGENCY BRAKE', rect_start_coordinate + 1, 1)

    def show_debug_text_on_display(self, text):
        if self.display is not None:
            self.display.setColor(0xFFFFFF)
            self.display.fillRectangle(0, 20, self.display.getWidth(), 10)
            self.display.setColor(0x000000)
            self.display.drawText(text, 1, 21)

    def start_devices(self):
        """Start the devices on the car and initialize objects like classifier."""
        # Start camera and the sensors:
        self.camera = self.getCamera(self.camera_device_name)
        if self.camera is not None:
            self.camera.enable(50)
        if self.DEBUG_DISPLAY:
            self.display = self.getDisplay(self.display_device_name)
        if self.display is not None:
            if self.camera is not None:
                self.display.attachCamera(self.camera)
        self.gps = self.getGPS(self.gps_device_name)
        if self.gps is not None:
            self.gps.enable(10)
        self.compass = self.getCompass(self.compass_device_name)
        if self.compass is not None:
            self.compass.enable(10)
        self.receiver = self.getReceiver(self.receiver_device_name)
        if self.receiver is not None:
            self.receiver.enable(10)
        self.emitter = self.getEmitter(self.emitter_device_name)
        # Start classifier, object tracker.
        self.classifier = Classifier(is_show_image=False)
        if self.is_online_classifier:
            self.classification_client = ClassificationClient()
            self.classification_client.establish_communication()
        else:
            self.classifier.start_classification_engine()
        self.obj_tracker = ObjectTracker()
        self.ground_truth_generator = GroundTruthGenerator()
        if self.camera is not None:
            self.ground_truth_generator.set_camera_parameters(self.camera.getWidth(),
                                                              self.camera.getHeight(),
                                                              self.camera.getFov())
        # Start the car engine
        self.start_car()

    def run(self):
        """Runs the controller."""
        self.start_devices()
        print("Devices Started.")
        sys.stdout.flush()
        debug_text = ''
        emergency_brake_until_ms = 0
        steering_maneuver_until_ms = 0
        steering_maneuver_direction = 'LEFT'
        slow_down_until_ms = 0
        is_tracker_set = False
        new_boxes = None
        object_distances = []
        detection_classes = []
        vhc_corners_dict = {}
        ped_corners_dict = {}
        print("Here we roll!")
        sys.stdout.flush()
        obj_total_perf_dict = {}
        perf_count = 0
        last_dist = 1000.0
        first_step = True
        target_lateral_pos = self.target_lat_pos
        while self.step() >= 0:
            if self.gps is not None:
                gps_speed = self.gps.getSpeed()
                current_self_position = self.gps.getValues()
                current_speed_m_s = max(np.linalg.norm(gps_speed), 0.0001)  # Against division by zero
            else:
                current_self_position = [0.0, 0.0, 0.0]
                current_speed_m_s = 0.0001
            if first_step:
                current_speed_m_s = 0.0
                first_step = False
            if self.receiver is not None:
                messages = self.contr_comm.receive_all_communication(self.receiver)
                command_list = self.contr_comm.extract_all_commands_from_message(messages)
            else:
                command_list = []
            vhc_pos_dict = self.contr_comm.get_all_vehicle_positions(command_list)
            ped_pos_dict = self.contr_comm.get_all_pedestrian_positions(command_list)
            vhc_rot_dict = self.contr_comm.get_all_vehicle_rotations(command_list)
            ped_rot_dict = self.contr_comm.get_all_pedestrian_rotations(command_list)
            vhc_corners_dict_temp = self.contr_comm.get_all_vehicle_box_corners(command_list)
            if len(vhc_corners_dict_temp) > 0:
                vhc_corners_dict = vhc_corners_dict_temp.copy()
            ped_corners_dict_temp = self.contr_comm.get_all_pedestrian_box_corners(command_list)
            if len(ped_corners_dict_temp) > 0:
                ped_corners_dict = ped_corners_dict_temp.copy()

            cycle_start_time = timeit.default_timer()
            #detection_boxes_to_track = []
            sim_time = self.get_sim_time()
            cur_sime_time_ms = int(round(1000 * sim_time))
            if cur_sime_time_ms % self.CLASSIFIER_PERIOD_MS == 0:
                if is_tracker_set:
                    # Before updating the objects in the tracker with the new image data,
                    # we want to track where the old objects are now.
                    tracker_object_distances = object_distances[:]
                    tracker_detection_classes = detection_classes[:]
                debug_text = 'Spd: {0:.1f} '.format(current_speed_m_s)

                # Do object detection:
                image_string = self.camera.getImage()
                np_image = np.fromstring(image_string, dtype=np.uint8)
                original_image = self.classifier.convert_data_to_image(np_image, \
                                                                       self.camera.getWidth(), \
                                                                       self.camera.getHeight())
                # First update the tracked object positions in the new frame. Then, set new objects to track.
                if is_tracker_set and tracker_object_distances and min(tracker_object_distances) < 70.0:
                    (new_boxes, old_boxes) = self.obj_tracker.get_tracking_results(original_image)
                else:
                    new_boxes = []
                    old_boxes = []
                if self.is_online_classifier:
                    (detection_boxes, detection_probs, detection_classes) = \
                        self.classification_client.get_classification_results(np_image, \
                                                                              self.camera.getWidth(), \
                                                                              self.camera.getHeight())
                else:
                    (_detection_image, (detection_boxes, detection_probs, detection_classes)) = \
                        self.classifier.do_object_detection(original_image, is_return_image=False)

                bbox_dict = self.ground_truth_generator.get_all_obj_bounding_boxes(1,
                                                                                   self.CAMERA_LOCAL_COORDINATES,
                                                                                   self.CAMERA_LOCAL_ROTATION,
                                                                                   vhc_pos_dict,
                                                                                   vhc_rot_dict,
                                                                                   vhc_corners_dict,
                                                                                   ped_pos_dict,
                                                                                   ped_rot_dict,
                                                                                   ped_corners_dict)
                for (obj_id, obj_class_text) in bbox_dict:
                    (truncation, gt_bbox) = bbox_dict[(obj_id, obj_class_text)]
                    if obj_class_text == 'Pedestrian':
                        obj_pos = ped_pos_dict[obj_id]
                        obj_class_label = self.classifier.PEDESTRIAN_CLASS_LABEL
                    else:
                        obj_pos = vhc_pos_dict[obj_id]
                        obj_class_label = self.classifier.CAR_CLASS_LABEL
                    self_pos = vhc_pos_dict[self.self_vhc_id]
                    obj_dist = np.linalg.norm(np.array(obj_pos) - np.array(self_pos))
                    if obj_dist < 70.0 and (0 < gt_bbox[0] - gt_bbox[2]/2 < self.camera.getWidth() \
                        or 0 < gt_bbox[0] + gt_bbox[2]/2 < self.camera.getWidth() \
                        or 0 < gt_bbox[0] < self.camera.getWidth())\
                        and \
                        (0 < gt_bbox[1] < self.camera.getHeight() \
                        or 0 < gt_bbox[1] + gt_bbox[3]/2 < self.camera.getHeight() \
                        or 0 < gt_bbox[1] - gt_bbox[3]/2 < self.camera.getHeight()) \
                        and truncation < 0.85:
                        (det_perf, det_iou, det_iou_ind) = iou_performance_for_object(gt_bbox, \
                                                                                        detection_boxes, \
                                                                                        detection_probs=detection_probs, \
                                                                                        prob_threshold=self.classifier.PROBABILITY_THRESHOLD, \
                                                                                        obj_class=obj_class_label, \
                                                                                        detection_classes=detection_classes)
                        if (obj_id, obj_class_text) in obj_total_perf_dict:
                            (total_perf, perf_count) = obj_total_perf_dict[(obj_id, obj_class_text)]
                        else:
                            total_perf = 0.0
                            perf_count = 0
                        total_perf += det_perf
                        perf_count += 1
                        obj_total_perf_dict[(obj_id, obj_class_text)] = (total_perf, perf_count)
                        message = self.contr_comm.generate_detection_box_perf_message(obj_id, obj_class_text, total_perf/perf_count)
                        self.emitter.send(message)
                        debug_text += 'Obj {}, {} Avg perf: {:.2f} '.format(obj_id, obj_class_text, total_perf/perf_count)
                    else:
                        if (obj_id, obj_class_text) in obj_total_perf_dict:
                            (total_perf, perf_count) = obj_total_perf_dict[(obj_id, obj_class_text)]
                        else:
                            total_perf = 100.0
                            perf_count = 1.0
                        message = self.contr_comm.generate_detection_box_perf_message(obj_id, obj_class_text, total_perf/perf_count)
                        self.emitter.send(message)
                # Naive collision avoidance:
                self.refresh_display_overlay()
                ped_count = 0
                car_count = 0
                object_distances = []
                object_sides = []
                collision_object_distance = 10000.0
                for det_obj_ind in range(len(detection_classes)):
                    self.show_object_on_display(detection_classes[det_obj_ind], detection_boxes[det_obj_ind], det_obj_ind, with_text=True)
                    det_x_pos = detection_boxes[det_obj_ind][0]
                    det_y_pos = detection_boxes[det_obj_ind][1]
                    det_width = detection_boxes[det_obj_ind][2]
                    det_height = detection_boxes[det_obj_ind][3]
                    dist = self.compute_distance(detection_classes[det_obj_ind], det_height)\
                           - self.CAMERA_TO_FRONT_DISTANCE
                    object_distances.append(dist)
                    is_in_scope = self.is_in_scope_triangle(det_x_pos + det_width/2.0, min(383, det_y_pos+det_height/2), detection_classes[det_obj_ind]) \
                                  or self.is_in_scope_triangle(det_x_pos - det_width/2.0, min(383, det_y_pos+det_height/2), detection_classes[det_obj_ind]) \
                                  or self.is_in_scope_triangle(det_x_pos, min(383, det_y_pos+det_height/2), detection_classes[det_obj_ind])
                    if is_in_scope:
                        object_sides.append('F')
                    elif det_x_pos < self.camera.getWidth()/2.0:
                        object_sides.append('L')
                    else:
                        object_sides.append('R')
                    if detection_classes[det_obj_ind] == Classifier.PEDESTRIAN_CLASS_LABEL:
                        ped_count += 1
                        debug_text += "Ped dist: {:.1f}".format(dist)
                        if is_in_scope and dist < collision_object_distance:
                            if (dist < 25.0 or (dist < 60.0 and dist/current_speed_m_s < 5.0)): #(dist < 45.0 and current_speed_m_s > 10.0) or (dist < 60.0 and current_speed_m_s > 15.0):
                                self.mark_critical_object_on_display(detection_boxes[det_obj_ind], 0xFF0000)
                                emergency_brake_until_ms = max(emergency_brake_until_ms, cur_sime_time_ms + self.MIN_EMERGENCY_BRAKE_DURATION_MS)
                                collision_object_distance = dist
                            if dist < 50.0 and dist/current_speed_m_s < self.EMERGENCY_STEERING_TTC:  # We will collide in 3.0 seconds
                                collision_object_distance = dist
                                steering_maneuver_until_ms = cur_sime_time_ms + self.MIN_STEERING_MANEUVER_MS
                                emergency_brake_until_ms = max(emergency_brake_until_ms, cur_sime_time_ms + self.MIN_STEERING_MANEUVER_MS)  # We do longer emergency braking because we may lose the object after we steer.
                                if det_x_pos < self.camera.getWidth()/2.0:  # on the left
                                    if current_self_position[0] > -1.75:  # There is space to steer to right
                                        steering_maneuver_direction = 'RIGHT'
                                    else:
                                        steering_maneuver_direction = 'LEFT'
                                else:
                                    if current_self_position[0] < 1.75:  # There is space to steer to left
                                        steering_maneuver_direction = 'LEFT'
                                    else:
                                        steering_maneuver_direction = 'RIGHT'
                        self.show_pedestrian_details_on_display(ped_count, dist, is_in_scope, det_obj_ind)
                    elif detection_classes[det_obj_ind] == Classifier.CAR_CLASS_LABEL:
                        car_count += 1
                        if is_in_scope and dist < collision_object_distance:
                            if (dist < 25.0 or (dist < 60.0 and dist/current_speed_m_s < 5.0)):  #(dist < 45.0 and current_speed_m_s > 10.0) or (dist < 60.0 and current_speed_m_s > 15.0):
                                collision_object_distance = dist
                                self.mark_critical_object_on_display(detection_boxes[det_obj_ind], 0xFF0000)
                                emergency_brake_until_ms = max(emergency_brake_until_ms, cur_sime_time_ms + self.MIN_EMERGENCY_BRAKE_DURATION_MS)
                            if dist < 50.0 and dist/current_speed_m_s < self.EMERGENCY_STEERING_TTC:  # We will collide in 3.0 seconds
                                collision_object_distance = dist
                                steering_maneuver_until_ms = cur_sime_time_ms + self.MIN_STEERING_MANEUVER_MS
                                emergency_brake_until_ms = max(emergency_brake_until_ms, cur_sime_time_ms + self.MIN_STEERING_MANEUVER_MS)  # We do longer emergency braking because we may lose the object after we steer.
                                if det_x_pos < self.camera.getWidth()/2.0:  # on the left
                                    if current_self_position[0] > -1.75:  # There is space to steer to right
                                        steering_maneuver_direction = 'RIGHT'
                                    else:
                                        steering_maneuver_direction = 'LEFT'
                                else:
                                    if current_self_position[0] < 1.75:  # There is space to steer to left
                                        steering_maneuver_direction = 'LEFT'
                                    else:
                                        steering_maneuver_direction = 'RIGHT'

                avg_obj_movements_x = []
                avg_obj_movements_y = []
                if new_boxes is not None:
                    for (box_ind, detection_box) in enumerate(new_boxes):
                        # This is not used for perf computation. Used for better matching detection boxes with tracker boxes.
                        (_max_perf, max_iou, max_iou_ind) = iou_performance_for_object(detection_box, detection_boxes, \
                                                                                    detection_probs=None, prob_threshold=0.0, \
                                                                                    obj_class=tracker_detection_classes[box_ind], \
                                                                                    detection_classes=detection_classes)
                        if max_iou > 30.0:
                            detection_box = detection_boxes[max_iou_ind][:]
                        self.show_object_on_display(tracker_detection_classes[box_ind], detection_box, box_ind, with_text=False)
                        old_box = old_boxes[box_ind]
                        # Doing an averaging for error filtering below:
                        dist = tracker_object_distances[box_ind]
                        if current_speed_m_s > 0.1:
                            time_for_me = dist / current_speed_m_s
                        else:
                            time_for_me = math.sqrt(2*dist / 1.5)  # If we are stopping, we want to make a best guess based on what if we accelerate constantly 1.5m/s^2.
                        num_time_steps_for_me = time_for_me / 0.010
                        if current_speed_m_s > 0.1:
                            time_for_me_short = max(0, (dist - 5.0)) / current_speed_m_s
                        else:
                            time_for_me_short = math.sqrt(2*max(0, (dist - 5.0)) / 1.5)  # If we are stopping, we want to make a best guess based on what if we accelerate constantly 1.5m/s^2.
                        num_time_steps_for_me = time_for_me / 0.010
                        num_time_steps_for_me_short = time_for_me_short / 0.010
                        obj_org_pos_x = detection_box[0]
                        obj_org_pos_y = detection_box[1]
                        obj_movement_x = detection_box[0] - old_box[0]
                        obj_movement_y = detection_box[1] - old_box[1]
                        if tracker_detection_classes[box_ind] == Classifier.PEDESTRIAN_CLASS_LABEL:
                            # If pedestrian move estimate is faster than 8 mph, limit it to 8 mph
                            obj_movement_x = max(-0.036, min(0.036, obj_movement_x))
                            obj_movement_y = max(-0.0036, min(0.0036, obj_movement_y)) # it can't move much on y axis even if it is going in the opposite direction as us
                        elif tracker_detection_classes[box_ind] == Classifier.CAR_CLASS_LABEL:
                            # If vehicle move estimate is faster than 60 mph, limit it to 60 mph
                            obj_movement_x = max(-0.27, min(0.27, obj_movement_x))
                            obj_movement_y = max(-0.027, min(0.027, obj_movement_y)) # it can't move much on y axis even if it is going in the opposite direction as us
                        if len(avg_obj_movements_x) <= box_ind:
                            avg_obj_movements_x.append(obj_movement_x)
                            avg_obj_movements_y.append(obj_movement_y)
                        else:
                            # New measurements are more valued:
                            avg_obj_movements_x[box_ind] = (avg_obj_movements_x[box_ind] + obj_movement_x) / 2.0
                            avg_obj_movements_y[box_ind] = (avg_obj_movements_y[box_ind] + obj_movement_y) / 2.0
                        total_obj_movement_x = avg_obj_movements_x[box_ind]*num_time_steps_for_me
                        total_obj_movement_y = avg_obj_movements_y[box_ind]*num_time_steps_for_me
                        total_obj_movement_x_short = avg_obj_movements_x[box_ind]*num_time_steps_for_me_short
                        total_obj_movement_y_short = avg_obj_movements_y[box_ind]*num_time_steps_for_me_short
                        end_obj_pos_x = obj_org_pos_x + total_obj_movement_x
                        end_obj_pos_y = obj_org_pos_y + total_obj_movement_y
                        end_obj_pos_x_short = obj_org_pos_x + total_obj_movement_x_short
                        end_obj_pos_y_short = obj_org_pos_y + total_obj_movement_y_short
                        currently_is_in_scope = self.is_in_scope_triangle(obj_org_pos_x + detection_box[2]/2.0, min(383, obj_org_pos_y+detection_box[3]/2), tracker_detection_classes[box_ind]) \
                                                or self.is_in_scope_triangle(obj_org_pos_x - detection_box[2]/2.0, min(383, obj_org_pos_y+detection_box[3]/2), tracker_detection_classes[box_ind]) \
                                                or self.is_in_scope_triangle(obj_org_pos_x, min(383, obj_org_pos_y+detection_box[3]/2), tracker_detection_classes[box_ind])
                        is_in_scope = self.is_in_scope_triangle(end_obj_pos_x, min(383, end_obj_pos_y+detection_box[3]/2), tracker_detection_classes[box_ind]) \
                                    or self.is_in_scope_triangle(end_obj_pos_x - detection_box[2]/2.0, min(383, end_obj_pos_y+detection_box[3]/2), tracker_detection_classes[box_ind]) \
                                    or self.is_in_scope_triangle(end_obj_pos_x + detection_box[2]/2.0, min(383, end_obj_pos_y+detection_box[3]/2), tracker_detection_classes[box_ind])
                        # is_in_scope_short is defining if the object will be in scope "before" we arrive the meeting point
                        is_in_scope_short = self.is_in_scope_triangle(end_obj_pos_x_short, min(383, end_obj_pos_y_short+detection_box[3]/2), tracker_detection_classes[box_ind]) \
                                    or self.is_in_scope_triangle(end_obj_pos_x_short - detection_box[2]/2.0, min(383, end_obj_pos_y_short+detection_box[3]/2), tracker_detection_classes[box_ind]) \
                                    or self.is_in_scope_triangle(end_obj_pos_x_short + detection_box[2]/2.0, min(383, end_obj_pos_y_short+detection_box[3]/2), tracker_detection_classes[box_ind])
                        if (is_in_scope or is_in_scope_short) and tracker_detection_classes[box_ind] == Classifier.PEDESTRIAN_CLASS_LABEL and dist < collision_object_distance:  # Tracker doesn't do well for parked cars.:
                            # Object will be in front of me when I arrive there.
                            # Normally object will be larger when I arrive, so width computation is incorrect but I ignore it for now.
                            if (dist < 25.0 or (dist < 60.0 and dist/current_speed_m_s < 5.0)):  # (dist < 45.0 and current_speed_m_s > 10.0) or (dist < 60.0 and current_speed_m_s > 15.0):
                                collision_object_distance = dist
                                emergency_brake_until_ms = max(emergency_brake_until_ms, cur_sime_time_ms + self.MIN_EMERGENCY_BRAKE_DURATION_MS)
                                self.mark_critical_object_on_display(detection_box, 0xFF0000)
                            if dist < 50.0 and time_for_me < self.EMERGENCY_STEERING_TTC:
                                if not (currently_is_in_scope and tracker_detection_classes[box_ind] == Classifier.PEDESTRIAN_CLASS_LABEL):
                                    collision_object_distance = dist
                                    steering_maneuver_until_ms = cur_sime_time_ms + self.MIN_STEERING_MANEUVER_MS
                                    emergency_brake_until_ms = max(emergency_brake_until_ms, cur_sime_time_ms + self.MIN_STEERING_MANEUVER_MS)  # We do longer emergency braking because we may lose the object after we steer.
                                    if total_obj_movement_x < 0:
                                        if current_self_position[0] > -1.75:  # There is space to steer to right
                                            steering_maneuver_direction = 'RIGHT'
                                        else:
                                            steering_maneuver_direction = 'LEFT'
                                    else:
                                        if current_self_position[0] < 1.75:  # There is space to steer to left
                                            steering_maneuver_direction = 'LEFT'
                                        else:
                                            steering_maneuver_direction = 'RIGHT'
                self.obj_tracker.set_tracked_objects(detection_boxes, original_image)
                is_tracker_set = True

            if self.compass is not None:
                self_bearing = get_bearing(self.compass)
            else:
                self_bearing = self.target_bearing
            angle_err = self.target_bearing - self_bearing
            while angle_err > math.pi:
                angle_err -= 2*math.pi
            while angle_err < -math.pi:
                angle_err += 2*math.pi
            closer_object_on_left = False
            closer_object_on_right = False
            if steering_maneuver_until_ms > cur_sime_time_ms:
                # Check if we will collide with something else if we steer. If so, cancel steering.
                for (obj_ind, obj_side) in enumerate(object_sides):
                    if ((obj_side == 'R') and 
                         object_distances[obj_ind] < collision_object_distance):
                        closer_object_on_right = True
                    if ((obj_side == 'L') and 
                         object_distances[obj_ind] < collision_object_distance):
                        closer_object_on_left = True
            if steering_maneuver_until_ms > cur_sime_time_ms:
                self.lateral_controller.k = 0.4
                if steering_maneuver_direction == 'RIGHT':
                    if closer_object_on_right and not closer_object_on_left:
                        steering_maneuver_direction = 'LEFT'
                else:
                    if closer_object_on_left and not closer_object_on_right:
                        steering_maneuver_direction = 'RIGHT'

                if steering_maneuver_direction == 'RIGHT':
                    if closer_object_on_right and closer_object_on_left:
                        if current_self_position[0] > 1.75:
                            target_lateral_pos = max(target_lateral_pos - self.LANE_WIDTH/2.0, -1.75)
                        else:
                            target_lateral_pos = max(target_lateral_pos - self.LANE_WIDTH/2.0, -3.75)
                    else:
                        if current_self_position[0] > 1.75:
                            target_lateral_pos = max(target_lateral_pos - self.LANE_WIDTH, -1.75)
                        else:
                            target_lateral_pos = max(target_lateral_pos - self.LANE_WIDTH, -3.75)
                else:
                    if closer_object_on_left and closer_object_on_right:
                        if current_self_position[0] < -1.75:
                            target_lateral_pos = min(target_lateral_pos + self.LANE_WIDTH/2.0, 1.75)
                        else:
                            target_lateral_pos = min(target_lateral_pos + self.LANE_WIDTH/2.0, 3.75)
                    else:
                        if current_self_position[0] < -1.75:
                            target_lateral_pos = min(target_lateral_pos + self.LANE_WIDTH, 1.75)
                        else:
                            target_lateral_pos = min(target_lateral_pos + self.LANE_WIDTH, 3.75)
            else:
                self.lateral_controller.k = 0.15
                target_lateral_pos = self.target_lat_pos
            steering = self.lateral_controller.compute(angle_err, current_self_position[0] - target_lateral_pos, current_speed_m_s)
            # Will do emergency brake for 2 frames if pedestrian was detected once
            if emergency_brake_until_ms > cur_sime_time_ms:
                throttle = -1.0
                is_in_emergency_brake = True
            elif slow_down_until_ms > cur_sime_time_ms:
                throttle = self.target_throttle / 2.0
            else:
                throttle = self.target_throttle
                is_in_emergency_brake = False
            self.set_throttle_and_steering_angle(throttle, steering)
            if debug_text != '':
                self.show_debug_text_on_display(debug_text)
            self.show_control_information_on_display(throttle, steering, is_in_emergency_brake)
            cycle_elapsed_time = timeit.default_timer() - cycle_start_time
            sys.stdout.flush()

        # Clean up
        if self.is_online_classifier:
            self.classification_client.close_communication()
            del self.classification_client
        del self.classifier
        del self.obj_tracker
        print("Bye!")
        sys.stdout.flush()
