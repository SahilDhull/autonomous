"""Defines ClassificationEvaluator class
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
import numpy as np
from Sim_ATAV.vehicle_control.base_controller.base_controller import BaseCarController
from Sim_ATAV.classifier.classifier_interface.classifier import Classifier
from Sim_ATAV.classifier.classifier_interface.ground_truth_generator import GroundTruthGenerator
from Sim_ATAV.common.controller_communication_interface import ControllerCommunicationInterface


def classify_camera_image(classifier, image_1D, width, height, return_image=False):
    (detection_boxes, detection_probs, detection_classes, detection_image, _original_image) = \
        classifier.do_object_detection_on_raw_data(image_1D, width, height, is_return_det_image=return_image, is_return_org_image=False)
    return detection_boxes, detection_probs, detection_classes, detection_image


class ClassificationEvaluator(BaseCarController):
    """ClassificationEvaluator class is a car controller class for Webots.
    It is used for experimenting basic interface with an object detection framework.
    When it detects a pedestrian in front of the car, it applies emergency braking,
    otherwise drives the car with given target throttle."""

    CAMERA_TO_FRONT_DISTANCE = 2.6  # 2.6 m is the distance from camera to Prius front face
    PED_DISTANCE_SCALE_FACTOR = 2793  # Focal length * known height for pedestrian
    CAR_DISTANCE_SCALE_FACTOR = 2483  # Focal length * known height for car
    # Defining a triangle with lines y1 = m1*x + b1 and y2 = m2*x+b2:
    CAR_FRONT_TRIANGLE_LINE1_M = -0.6  # Line 1 m for front triangle.
    CAR_FRONT_TRIANGLE_LINE1_B = 526  # Line 1 b for front triangle.
    CAR_FRONT_TRIANGLE_LINE2_M = 0.6  # Line 2 m for front triangle.
    CAR_FRONT_TRIANGLE_LINE2_B = -202  # Line 2 b for front triangle.

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

    BOX_NAME_DICT = {Classifier.CAR_CLASS_LABEL:CAR_BOX_TEXT,
                     Classifier.PEDESTRIAN_CLASS_LABEL:PEDESTRIAN_BOX_TEXT,
                     Classifier.CYCLIST_CLASS_LABEL:CYCLIST_BOX_TEXT}

    def __init__(self, controller_parameters):
        (car_model, target_throttle) = controller_parameters
        BaseCarController.__init__(self, car_model)
        self.camera_device_name = "camera"
        self.camera = None
        self.display_device_name = 'display'
        self.display = None
        self.receiver_device_name = 'receiver'
        self.receiver = None
        self.target_throttle = float(target_throttle)
        self.ground_truth_generator = GroundTruthGenerator()
        self.contr_comm = ControllerCommunicationInterface()
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

    def is_in_scope_triangle(self, pixel_x, pixel_y):
        """Check if object is in front of the car."""
        in_triangle = False
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

    def show_object_on_display(self, obj_class, detection_box, obj_ind):
        if self.display is not None:
            self.display.setColor(self.BOX_COLOR_DICT[obj_class])
            # detection_box array: [x, y, w, h]
            # Detection box from SqueezeDet has x and y as the center of image.
            # Webots expects x and y as the top left corner of the image.
            self.display.drawRectangle(int(detection_box[0] - detection_box[2]/2.0), \
                                       int(detection_box[1] - detection_box[3]/2.0), \
                                       int(detection_box[2]), \
                                       int(detection_box[3]))
            self.display.drawText(self.BOX_NAME_DICT[obj_class] + '_' + str(obj_ind), \
                                  int(detection_box[0] - detection_box[2]/2.0) + 1, \
                                  int(detection_box[1] - detection_box[3]/2.0) + 1)

    def show_pedestrian_details_on_display(self, ped_count, dist, is_pedestrian_in_scope, obj_ind):
        if self.display is not None:
            self.display.setColor(0xFFFFFF)
            self.display.drawText('Ped_' + str(obj_ind) + ' Dist: ' + str(int(dist)) + ' Front?: ' + \
                                  str(is_pedestrian_in_scope), 200*(ped_count - 1) + 1, 11)

    def show_control_information_on_display(self, throttle, steering, is_emergency_brake=False):
        if self.display is not None:
            self.display.setColor(0x00FF00)
            self.display.fillRectangle(0, 0, self.display.getWidth(), 10)
            self.display.setColor(0x0000FF)
            self.display.drawText('Throttle: ' + str(throttle) + ' Steering: ' + str(steering), 1, 1)
            if is_emergency_brake:
                rect_start_coordinate = int(self.display.getWidth() / 2.0)
                self.display.setColor(0xFF0000)
                self.display.fillRectangle(rect_start_coordinate, 0, self.display.getWidth()-rect_start_coordinate, 10)
                self.display.setColor(0xFFFFFF)
                self.display.drawText('EMERGENCY BRAKE', rect_start_coordinate + 1, 1)

    def run(self):
        """Runs the controller."""
        classifier = Classifier(is_show_image=False)
        # Start camera and the car engine:
        self.camera = self.getCamera(self.camera_device_name)
        if self.camera is not None:
            self.camera.enable(10)
            self.ground_truth_generator.set_camera_parameters(self.camera.getWidth(),
                                                              self.camera.getHeight(),
                                                              self.camera.getFov())
        self.display = self.getDisplay(self.display_device_name)
        if self.display is not None:
            if self.camera is not None:
                self.display.attachCamera(self.camera)
        self.receiver = self.getReceiver(self.receiver_device_name)
        if self.receiver is not None:
            self.receiver.enable(10)
            print('Receiver Enabled!')
        self.start_car()

        print("Classification Evaluator Started!")
        sys.stdout.flush()
        classifier.start_classification_engine()
        print('Classification Engine Started!')
        sys.stdout.flush()
        in_emergency_brake = 0
        counter = 0
        vhc_corners_dict = {}
        ped_corners_dict = {}
        while self.step() >= 0:
            sys.stdout.flush()
            # self.refresh_display_overlay()

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

            counter += 1
            if counter == 10001:
                counter = 1
            if counter % 5 == 1:
                # Read image. Format: 1st line B,G,R,A values, 2nd line B,G,R,A values ... (1D string)
                image_string = self.camera.getImage()

                # Convert image to numpy array of BGR values:
                image_1D = np.delete(np.fromstring(image_string, dtype=np.uint8),\
                                     slice(3, len(image_string), 4))

                # Do object detection:
                (detection_boxes, detection_probs, detection_classes, detection_image) = \
                    classify_camera_image(classifier, image_1D, self.camera.getWidth(), self.camera.getHeight(), return_image=False)

                # Super naive pedestrian collision avoidance:
                self.refresh_display_overlay()
                ped_count = 0
                for det_obj_ind in range(len(detection_classes)):
                    self.show_object_on_display(detection_classes[det_obj_ind], detection_boxes[det_obj_ind], det_obj_ind)
                    if detection_classes[det_obj_ind] == Classifier.PEDESTRIAN_CLASS_LABEL:
                        ped_count += 1
                        pedestrian_x_pos = detection_boxes[det_obj_ind][0]
                        pedestrian_y_pos = detection_boxes[det_obj_ind][1]
                        pedestrian_height = detection_boxes[det_obj_ind][3]
                        dist = self.compute_distance(detection_classes[det_obj_ind], pedestrian_height)\
                               - self.CAMERA_TO_FRONT_DISTANCE

                        is_pedestrian_in_scope = self.is_in_scope_triangle(pedestrian_x_pos, pedestrian_y_pos)
                        if is_pedestrian_in_scope and dist < 50.0:
                            in_emergency_brake = 25
                        self.show_pedestrian_details_on_display(ped_count, dist, is_pedestrian_in_scope, det_obj_ind)

                detection_box_list = \
                    self.ground_truth_generator.get_all_vhc_and_ped_detection_boxes(1,
                                                                                    self.CAMERA_LOCAL_COORDINATES,
                                                                                    self.CAMERA_LOCAL_ROTATION,
                                                                                    vhc_pos_dict,
                                                                                    vhc_rot_dict,
                                                                                    vhc_corners_dict,
                                                                                    ped_pos_dict,
                                                                                    ped_rot_dict,
                                                                                    ped_corners_dict)
                for detection_box in detection_box_list:
                    if detection_box[0] == 'Car':
                        self.display.setColor(0xFFFF00)
                        label = 'CAR{}'.format(detection_box[1])
                    else:
                        self.display.setColor(0x000000)
                        label = 'PED{}'.format(detection_box[1])
                    self.display.drawRectangle(detection_box[3], detection_box[4],
                                               detection_box[5] - detection_box[3],
                                               detection_box[6] - detection_box[4])
                    self.display.drawText(label, detection_box[3], detection_box[4])

            # Will do emergency brake for 5 frames if pedestrian was detected once
            if in_emergency_brake > 0:
                throttle = -1.0
                steering = 0.0
                in_emergency_brake = in_emergency_brake - 1
            else:
                throttle = self.target_throttle
                steering = 0.0
            self.set_throttle_and_steering_angle(throttle, steering)
            self.show_control_information_on_display(throttle, steering, in_emergency_brake)
