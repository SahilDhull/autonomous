"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation.sensor_object import SensorObject
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation.radar_object import RadarObject


def radar_list_to_sensor_list(radar_objects, cur_time_ms):
    """Convert a list of radar targets into sensor objects."""
    detected_objects = []
    for radar_object in radar_objects:
        detected_objects.append(SensorObject(radar_to_object_class(radar_object.object_class)))
        detected_objects[-1].set_object_speed_m_s(-radar_object.radar_target.speed)
        detected_objects[-1].set_object_position(radar_object.relative_position[:])
        detected_objects[-1].sensor_recorded_position = radar_object.relative_position[:]
        detected_objects[-1].set_detection_time(cur_time_ms)
        detected_objects[-1].set_object_seen_by_sensor(SensorObject.SENSOR_RADAR)
        detected_objects[-1].set_aux_sensor_data(SensorObject.SENSOR_RADAR, radar_object)
    return detected_objects


def radar_to_object_class(radar_class):
    """Convert from radar object class to sensor object class"""
    if radar_class == RadarObject.OBJECT_CAR:
        object_class = SensorObject.OBJECT_CAR
    elif radar_class == RadarObject.OBJECT_BIKE:
        object_class = SensorObject.OBJECT_BIKE
    elif radar_class == RadarObject.OBJECT_PEDESTRIAN:
        object_class = SensorObject.OBJECT_PEDESTRIAN
    else:
        object_class = SensorObject.OBJECT_UNKNOWN
    return object_class
