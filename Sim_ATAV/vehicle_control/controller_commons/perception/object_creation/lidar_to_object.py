"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation.sensor_object import SensorObject
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation.lidar_object import LidarObject


def lidar_list_to_sensor_list(lidar_objects, cur_time_ms):
    """Convert a list of radar targets into sensor objects."""
    detected_objects = []
    for lidar_object in lidar_objects:
        detected_objects.append(SensorObject(lidar_to_object_class(lidar_object.object_class)))
        detected_objects[-1].set_object_speed_m_s(0.0)
        detected_objects[-1].set_object_position(lidar_object.relative_position[:])
        detected_objects[-1].sensor_recorded_position = lidar_object.relative_position[:]
        detected_objects[-1].set_detection_time(cur_time_ms)
        detected_objects[-1].set_object_seen_by_sensor(SensorObject.SENSOR_LIDAR)
        detected_objects[-1].set_aux_sensor_data(SensorObject.SENSOR_LIDAR, lidar_object)
    return detected_objects


def lidar_to_object_class(lidar_class):
    """Convert from lidar object class to sensor object class"""
    if lidar_class == LidarObject.OBJECT_CAR:
        object_class = SensorObject.OBJECT_CAR
    elif lidar_class == LidarObject.OBJECT_BIKE:
        object_class = SensorObject.OBJECT_BIKE
    elif lidar_class == LidarObject.OBJECT_PEDESTRIAN:
        object_class = SensorObject.OBJECT_PEDESTRIAN
    elif lidar_class == LidarObject.OBJECT_TRUCK:
        object_class = SensorObject.OBJECT_TRUCK
    else:
        object_class = SensorObject.OBJECT_UNKNOWN
    return object_class
