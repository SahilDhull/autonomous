"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


class SensorObject(object):
    """SensorObject defines features of the objects returned from the sensor suite."""
    OBJECT_CAR = 1
    OBJECT_PEDESTRIAN = 2
    OBJECT_BIKE = 3
    OBJECT_BUS = 4
    OBJECT_TRUCK = 5
    OBJECT_UNKNOWN = 100

    SENSOR_CAMERA = 'camera'
    SENSOR_LIDAR = 'lidar'
    SENSOR_RADAR = 'radar'

    NO_RISK = 0
    CAUTION = 5
    RISKY = 10
    HIGH_RISK = 15

    def __init__(self, object_type=None):
        self.object_type = object_type
        self.object_speed_m_s = 0.0
        self.object_direction = [0, 0]
        self.object_position = [0, 0]
        self.object_yaw_angle = 0.0
        self.object_yaw_rate = 0.0
        self.sensor_recorded_position = None
        self.sensor_recorded_direction = None
        self.global_position = None
        self.sensor_recorded_global_position = None
        self.global_yaw_angle = None
        self.global_yaw_rate = None
        self.global_speed = None
        self.is_first_detection = True
        self.detection_time = 0
        self.update_time = 0
        self.seen_by_sensor_dict = {self.SENSOR_CAMERA:False,
                                    self.SENSOR_LIDAR:False,
                                    self.SENSOR_RADAR:False}
        self.sensor_aux_data_dict = {}
        self.tracker = None
        self.history = []
        self.future = []
        self.aux_data = {}
        self.is_old_object = False
        self.risk_level = self.NO_RISK

    def set_object_type_classifier(self, classifier_object_type, classifier_class):
        """Set object type using Classifier types."""
        if classifier_object_type == classifier_class.CAR_CLASS_LABEL:
            self.object_type = self.OBJECT_CAR
        elif classifier_object_type == classifier_class.PEDESTRIAN_CLASS_LABEL:
            self.object_type = self.OBJECT_PEDESTRIAN
        elif classifier_object_type == classifier_class.CYCLIST_CLASS_LABEL:
            self.object_type = self.OBJECT_BIKE
        else:
            self.object_type = self.OBJECT_UNKNOWN

    def set_object_speed_m_s(self, object_speed_m_s):
        """Set detected speed for the object."""
        self.object_speed_m_s = object_speed_m_s

    def set_object_direction(self, object_direction):
        """Set detected object motion direction."""
        self.object_direction = object_direction[:]

    def set_object_position(self, object_position):
        """Set detected object relative position."""
        self.object_position = object_position

    def set_object_seen_by_sensor(self, sensor_type, is_seen=True):
        """Set if the object was seen by a particular sensor."""
        self.seen_by_sensor_dict[sensor_type] = is_seen

    def set_aux_sensor_data(self, sensor_type, aux_data=None):
        """Set auxiliary data from a particular sensor."""
        self.sensor_aux_data_dict[sensor_type] = aux_data

    def set_detection_time(self, detection_time):
        """Set last detected time of the object."""
        self.detection_time = detection_time
