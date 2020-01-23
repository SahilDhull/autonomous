"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from Sim_ATAV.vehicle_control.controller_commons.perception.sensing.camera_detection import CameraDetection
from Sim_ATAV.vehicle_control.controller_commons.perception.sensing.radar_detection import RadarDetection
from Sim_ATAV.vehicle_control.controller_commons.perception.sensing.lidar_detection import LidarDetection
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation import camera_to_object
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation import radar_to_object
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation import lidar_to_object


class ObjectDetection(object):
    def __init__(self):
        self.available_sensors = []
        self.camera_objects = []
        self.radar_objects = []
        self.lidar_objects = []
        self.detected_objects = []
        self.radar_targets = []
        self.lidar_point_clouds = []
        self.lidar_clusters = []
        self.has_camera = False
        self.has_radar = False
        self.has_lidar = False
        self.is_camera_read = False
        self.is_radar_read = False
        self.is_lidar_read = False

    def register_sensor(self, sensor_detector, sensor_period):
        self.available_sensors.append((sensor_detector, sensor_period))
        if isinstance(sensor_detector, CameraDetection):
            self.has_camera = True
        if isinstance(sensor_detector, RadarDetection):
            self.has_radar = True
        if isinstance(sensor_detector, LidarDetection):
            self.has_lidar = True

    def detect_objects(self, cur_time_ms):
        self.clear_detections()
        for sensor in self.available_sensors:
            sensor_period = sensor[1]
            if cur_time_ms % sensor_period == 0:
                sensor_detector = sensor[0]
                if isinstance(sensor_detector, CameraDetection):
                    (camera_objects, is_read) = sensor_detector.read_camera_and_find_objects()
                    if is_read:
                        self.camera_objects = camera_to_object.camera_list_to_sensor_list(camera_objects, cur_time_ms)
                        self.detected_objects = self.detected_objects + self.camera_objects
                        self.is_camera_read = True
                elif isinstance(sensor_detector, RadarDetection):
                    (radar_objects, radar_targets, is_read) = sensor_detector.read_radar_and_find_objects()
                    if is_read:
                        self.radar_objects = radar_to_object.radar_list_to_sensor_list(radar_objects, cur_time_ms)
                        self.detected_objects = self.detected_objects + self.radar_objects
                        self.radar_targets = radar_targets
                        self.is_radar_read = True
                elif isinstance(sensor_detector, LidarDetection):
                    (lidar_objects, lidar_point_cloud, lidar_clusters, is_read) = \
                        sensor_detector.read_lidar_and_find_objects()
                    if is_read:
                        self.lidar_objects = lidar_to_object.lidar_list_to_sensor_list(lidar_objects, cur_time_ms)
                        self.detected_objects = self.detected_objects + self.lidar_objects
                        self.lidar_clusters = lidar_clusters
                        self.lidar_point_clouds = lidar_point_cloud
                        self.is_lidar_read = True
        return self.detected_objects

    def get_detections(self):
        return self.detected_objects

    def clear_detections(self):
        self.detected_objects = []
        self.is_camera_read = False
        self.is_radar_read = False
        self.is_lidar_read = False
