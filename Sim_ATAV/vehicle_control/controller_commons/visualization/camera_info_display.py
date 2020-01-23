"""Defines CameraInfoDisplay class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

from Sim_ATAV.classifier.classifier_interface.classifier import Classifier


class CameraInfoDisplay(object):
    """CameraInfoDisplay class handles common functions for overlaying information on camera image."""
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

    def __init__(self, display_device):
        self.display = display_device
        if self.display is not None:
            self.display_width = self.display.getWidth()
            self.display_height = self.display.getHeight()
        else:
            self.display_width = 0
            self.display_height = 0

    def attach_camera(self, camera_device):
        """Attach a camera device to this display.
        The camera image from this device will always be on the display.
        Later, we will overlay information on top the camera image."""
        if self.display is not None and camera_device is not None:
            self.display.attachCamera(camera_device)

    def refresh_display_overlay(self):
        """Removes everything from the overlay on the camera image."""
        if self.display is not None:
            # Remove objects:
            self.display.setAlpha(0.0)
            self.display.fillRectangle(0, 0, self.display.getWidth(), self.display.getHeight())
            self.display.setAlpha(1.0)

    def show_object_detection_box(self, obj_class, detection_box, obj_ind=0, show_label_text=True, is_tracker_detection=False):
        """Shows object rectangle."""
        if self.display is not None:
            if is_tracker_detection:
                self.display.setColor(0xFFFF00)
            else:
                self.display.setColor(self.BOX_COLOR_DICT[obj_class])
            # detection_box array: [x, y, w, h]
            # Detection box from SqueezeDet has x and y as the center of image.
            # Webots expects x and y as the top left corner of the image.
            box_top_left_x = int(detection_box[Classifier.DET_BOX_X_IND] - detection_box[Classifier.DET_BOX_WIDTH_IND]/2.0)
            box_top_left_y = int(detection_box[Classifier.DET_BOX_Y_IND] - detection_box[Classifier.DET_BOX_HEIGHT_IND]/2.0)
            self.display.drawRectangle(box_top_left_x,
                                       box_top_left_y,
                                       int(detection_box[Classifier.DET_BOX_WIDTH_IND]),
                                       int(detection_box[Classifier.DET_BOX_HEIGHT_IND]))
            if show_label_text:
                self.display.drawText(self.BOX_NAME_DICT[obj_class] + '_' + str(obj_ind),
                                      box_top_left_x + 1,
                                      box_top_left_y + 1)

    def mark_critical_object(self, detection_box, color=0xFF0000):
        """Marks the object which was identified as critical."""
        if self.display is not None:
            self.display.setColor(color)
            self.display.setAlpha(0.7)
            # detection_box array: [x, y, w, h]
            # Detection box from SqueezeDet has x and y as the center of image.
            # Webots expects x and y as the top left corner of the image.
            box_top_left_x = int(detection_box[Classifier.DET_BOX_X_IND] - detection_box[Classifier.DET_BOX_WIDTH_IND]/2.0)
            box_top_left_y = int(detection_box[Classifier.DET_BOX_Y_IND] - detection_box[Classifier.DET_BOX_HEIGHT_IND]/2.0)
            if detection_box[Classifier.DET_BOX_HEIGHT_IND] < 20:
                self.display.fillRectangle(box_top_left_x,
                                           box_top_left_y,
                                           int(detection_box[Classifier.DET_BOX_WIDTH_IND]),
                                           int(detection_box[Classifier.DET_BOX_HEIGHT_IND] - 10))
            else:
                self.display.fillRectangle(box_top_left_x,
                                           box_top_left_y + 10,
                                           int(detection_box[Classifier.DET_BOX_WIDTH_IND]),
                                           int(detection_box[Classifier.DET_BOX_HEIGHT_IND] - 10))
            self.display.setAlpha(1.0)

    def show_control_information(self, throttle, steering, is_emergency_brake=False, control_mode=None):
        """Show the vehicle control (throttle/brake) information on the display."""
        if self.display is not None:
            self.display.setColor(0x00FF00)
            self.display.fillRectangle(0, 10, self.display.getWidth(), 10)
            self.display.setColor(0x0000FF)
            if control_mode is None:
                self.display.drawText('Throttle: {:.1f} Steering: {:.2f}'.format(throttle, steering), 1, 11)
            else:
                self.display.drawText('Throttle: {:.1f} Steering: {:.2f} Mode: {}'.format(throttle, steering, control_mode), 1, 11)
            if is_emergency_brake or throttle < -0.99:
                rect_start_coordinate = int(self.display.getWidth() / 2.0)
                self.display.setColor(0xFF0000)
                self.display.fillRectangle(rect_start_coordinate, 10, self.display.getWidth()-rect_start_coordinate, 10)
                self.display.setColor(0xFFFFFF)
                self.display.drawText('EMERGENCY BRAKE', rect_start_coordinate + 1, 11)

    def show_debug_text(self, text):
        """Show an information text."""
        if self.display is not None:
            self.display.setColor(0xFFFFFF)
            self.display.fillRectangle(0, 0, self.display.getWidth(), 10)
            self.display.setColor(0x000000)
            self.display.drawText(text, 1, 1)
