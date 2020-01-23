"""Defines SimpleClassifierControl class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import numpy as np
from Sim_ATAV.vehicle_control.base_controller.base_controller import BaseCarController
from Sim_ATAV.classifier.classifier_interface.classifier import Classifier


class SimpleClassifierControl(BaseCarController):
    """SimpleClassifierControl class is a car controller class for Webots.
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

    def __init__(self, controller_parameters):
        (car_model, target_throttle) = controller_parameters
        BaseCarController.__init__(self, car_model)
        self.camera_device_name = "camera"
        self.camera = None
        self.target_throttle = float(target_throttle)
        print("simple_classifier Initialized: {}, {}".format(car_model, self.target_throttle))

    def compute_distance(self, object_class, pixel_height):
        """Convert detected object pixel height to distance to the object."""
        if pixel_height < 1:
            pixel_height = 1
        if object_class == "pedestrian":
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

    def run(self):
        """Runs the controller."""
        classifier = Classifier(is_show_image=True)
        # Start camera and the car engine:
        self.camera = self.getCamera(self.camera_device_name)
        if self.camera is not None:
            self.camera.enable(16)
        self.start_car()

        print("Here we roll!")
        classifier.start_classification_engine()
        vehicle_alive = True
        while vehicle_alive:
            self.step()
            # Read image. Format: 1st line B,G,R,A values, 2nd line B,G,R,A values ... (1D string)
            image_string = self.camera.getImage()
            # Convert image to numpy array of BGR values:
            image_1D = np.delete(np.fromstring(image_string, dtype=np.uint8),
                                 slice(3, len(image_string), 4))

            # Do object detection:
            width = self.camera.getWidth()
            height = self.camera.getHeight()
            (detection_boxes, detection_probs, detection_classes, _detection_image, _original_image) = \
                classifier.do_object_detection_on_raw_data(image_1D, width, height, is_return_det_image=False,
                                                           is_return_org_image=False)


            # Super naive pedestrian collision avoidance:
            for det_obj_ind in range(len(detection_classes)):
                object_x_pos = detection_boxes[det_obj_ind][0]
                object_y_pos = detection_boxes[det_obj_ind][1]
                object_height = detection_boxes[det_obj_ind][3]
                if detection_classes[det_obj_ind] == classifier.PEDESTRIAN_CLASS_LABEL:
                    dist = self.compute_distance("pedestrian", object_height)\
                           - self.CAMERA_TO_FRONT_DISTANCE
                    print('pedestrian height: {}, Distance: {}'.format(object_height, dist))
                    print('pedestrian x_pos: {}, y_pos: {}'.format(object_x_pos, object_y_pos))
                    sys.stdout.flush()
                elif detection_classes[det_obj_ind] == classifier.CAR_CLASS_LABEL:
                    dist = self.compute_distance("car", object_height)\
                           - self.CAMERA_TO_FRONT_DISTANCE
                    print('car height: {}, Distance: {}'.format(object_height, dist))
                    print('car x_pos: {}, y_pos: {}'.format(object_x_pos, object_y_pos))
                    sys.stdout.flush()

            self.set_throttle_and_steering_angle(self.target_throttle, 0.0)
