"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation.camera_object import CameraObject
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation.sensor_object import SensorObject


def camera_list_to_sensor_list(camera_objects, cur_time_ms):
    """Convert a list of camera objects into sensor objects."""
    detected_objects = []
    for camera_object in camera_objects:
        detected_objects.append(SensorObject(camera_to_sensor_class(camera_object.object_class)))
        detected_objects[-1].set_object_speed_m_s(0.0)
        detected_objects[-1].set_object_position(camera_object.relative_position[:])
        detected_objects[-1].sensor_recorded_position = camera_object.relative_position[:]
        detected_objects[-1].set_detection_time(cur_time_ms)
        detected_objects[-1].set_object_seen_by_sensor(SensorObject.SENSOR_CAMERA)
        detected_objects[-1].set_aux_sensor_data(SensorObject.SENSOR_CAMERA, camera_object)
    return detected_objects


def camera_to_sensor_class(camera_class):
    """Convert from camera object class to sensor object class"""
    if camera_class == CameraObject.OBJECT_CAR:
        object_class = SensorObject.OBJECT_CAR
    elif camera_class == CameraObject.OBJECT_BIKE:
        object_class = SensorObject.OBJECT_BIKE
    elif camera_class == CameraObject.OBJECT_PEDESTRIAN:
        object_class = SensorObject.OBJECT_PEDESTRIAN
    else:
        object_class = SensorObject.OBJECT_UNKNOWN
    return object_class
