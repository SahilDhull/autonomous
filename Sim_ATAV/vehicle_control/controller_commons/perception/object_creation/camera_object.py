"""Defines CameraObject class.
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


class CameraObject(object):
    OBJECT_CAR = 0
    OBJECT_PEDESTRIAN = 1
    OBJECT_BIKE = 2

    """CameraObject class defines features of the object detected by camera."""
    def __init__(self, relative_position=None, object_class=None, detection_box=None, class_probability=0,
                 classifier='squeezeDet'):
        self.relative_position = relative_position
        self.object_class = self.object_class_from_classifier(object_class, classifier)
        self.detection_box = detection_box
        self.class_probability = class_probability
        self.added_by_tracker = False
        self.tracker_aux_data = {}

    def object_class_from_classifier(self, object_class, classifier):
        """Converts classifier object class to camera object class.
        The argument classifier can be a string or the classifier object itself."""
        if classifier == 'squeezeDet':
            if object_class == 0:
                object_class_camera = self.OBJECT_CAR
            elif object_class == 1:
                object_class_camera = self.OBJECT_PEDESTRIAN
            else:
                object_class_camera = self.OBJECT_BIKE
        elif classifier is not None:  # here, classifier is an object from Classifier class
            if object_class == classifier.CAR_CLASS_LABEL:
                object_class_camera = self.OBJECT_CAR
            elif object_class == classifier.PEDESTRIAN_CLASS_LABEL:
                object_class_camera = self.OBJECT_PEDESTRIAN
            else:
                object_class_camera = self.OBJECT_BIKE
        else:
            object_class_camera = object_class
        return object_class_camera
