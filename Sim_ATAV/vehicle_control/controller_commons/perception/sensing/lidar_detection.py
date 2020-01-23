"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from Sim_ATAV.vehicle_control.controller_commons import lidar_tools
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation.lidar_object import LidarObject


class LidarDetection(object):
    def __init__(self, lidar_device=None, lidar_relative_pos=(0.0, 0.0), lidar_layers=None):
        self.lidar = lidar_device
        self.lidar_relative_position = [lidar_relative_pos[0], lidar_relative_pos[1]]
        self.lidar_layers = [] if lidar_layers is None else lidar_layers[:]

    def read_lidar_and_find_objects(self):
        """Reads lidar layers and creates a list of detected objects."""
        lidar_point_cloud = []
        lidar_detected_objects = []
        lidar_clusters = []
        is_read = False

        if self.lidar is not None:
            is_read = True
            for layer_id in self.lidar_layers:
                lidar_point_cloud = lidar_point_cloud + self.lidar.getLayerPointCloud(layer_id)
            (lidar_detected_objects, lidar_clusters) = \
                self.find_objects_in_point_cloud(point_cloud=lidar_point_cloud)
        return lidar_detected_objects, lidar_point_cloud, lidar_clusters, is_read

    def find_objects_in_point_cloud(self, point_cloud):
        """Converts point cloud to x,y,z-only data, clusters it and returns a list of detected objects."""
        xyz_cloud = lidar_tools.convert_point_cloud_to_xyz(point_cloud, max_range=100)
        if len(xyz_cloud) > 10:
            pc_cluster_labels = lidar_tools.cluster_point_cloud(xyz_cloud)
            lidar_clusters = lidar_tools.filter_clustered_points(pc_cluster_labels, xyz_cloud)
        else:
            lidar_clusters = []
        lidar_detected_objects = []
        for cluster in lidar_clusters:
            position = [(cluster.min_x + cluster.max_x)/2.0 + self.lidar_relative_position[0],
                        (cluster.min_y + cluster.max_y)/2.0 + self.lidar_relative_position[1]]
            lidar_detected_objects.append(LidarObject(lidar_cluster=cluster,
                                                      relative_position=position[:],
                                                      object_class=self.get_lidar_object_class(cluster)))
        return lidar_detected_objects, lidar_clusters

    def get_lidar_object_class(self, lidar_cluster):
        """Estimate object type from Lidar point cloud cluster."""
        length_x = lidar_cluster.max_x - lidar_cluster.min_x
        length_y = lidar_cluster.max_y - lidar_cluster.min_y
        area = length_x * length_y
        if length_x > 5.0 or length_y > 5.0 or area > 12.0:
            object_class = LidarObject.OBJECT_TRUCK
        elif length_x > 1.2 or length_y > 1.2 or area > 2.0:
            object_class = LidarObject.OBJECT_CAR
        else:
            object_class = LidarObject.OBJECT_PEDESTRIAN
        return object_class
