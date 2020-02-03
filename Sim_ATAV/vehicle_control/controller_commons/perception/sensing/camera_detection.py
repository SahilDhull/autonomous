"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import numpy as np
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation.camera_object import CameraObject


class CameraDetection(object):
    def __init__(self, camera_device=None, classifier=None, cam_relative_pos=(0.0, 0.0)):
        self.camera = camera_device
        self.classifier = classifier
        self.camera_relative_position = [cam_relative_pos[0], cam_relative_pos[1]]

    def read_camera_and_find_objects(self):
        """Reads Camera and uses object detection and classification to create a list of detected objects."""
        camera_detected_objects = []
        is_read = False
        if self.camera is not None and self.classifier is not None:
            is_read = True
            camera_raw_image = self.camera.getImage()
            sensor_camera_image = self.classifier.convert_data_to_image(np.fromstring(camera_raw_image,
                                                                                      dtype=np.uint8),
                                                                        self.camera.getWidth(),
                                                                        self.camera.getHeight())
            # print(sensor_camera_image)
            
            # Neural Network Object Detection and Classification:
            (_detection_image, (detection_boxes, detection_probs, detection_classes)) = \
                self.classifier.do_object_detection(sensor_camera_image, is_return_image=False)
            for (obj_ind, detection_box) in enumerate(detection_boxes):
                position = self.classifier.box_to_relative_position(detection_classes[obj_ind],
                                                                    detection_box,
                                                                    self.camera.getWidth())
                position = [position[0] + self.camera_relative_position[0],
                            position[1] + self.camera_relative_position[1]]
                camera_detected_objects.append(CameraObject(relative_position=position,
                                                            object_class=detection_classes[obj_ind],
                                                            detection_box=detection_box,
                                                            class_probability=detection_probs[obj_ind],
                                                            classifier=self.classifier))

        return camera_detected_objects, is_read
