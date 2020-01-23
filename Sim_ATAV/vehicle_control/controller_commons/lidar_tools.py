"""Defines LidarTools class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import numpy as np
from sklearn.cluster import DBSCAN


class LidarCluster(object):
    """A structure to keep Lidar point cloud cluster information."""
    def __init__(self, cluster_points=None, min_x=0, max_x=0, min_y=0, max_y=0):
        self.cluster_points = cluster_points
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y


def convert_point_cloud_to_xyz(point_cloud, max_range=70, front_points_only=True):
    """Converts lidar point cloud data into x,y,z coordinates only
    (just extracts that already available information)."""
    xyz = [[pt.x, pt.y, pt.z] for pt in point_cloud]
    # pp[1] > -1.0 for removing points too low (pavements, road etc.)
    # pp[2] > 0 for only considering point in front of the vehicle.
    if front_points_only:
        xyz_mask = [True if (np.linalg.norm(pp) < max_range and pp[1] > -1.0 and pp[2] > 0) else False for pp in xyz]
    else:
        xyz_mask = [True if (np.linalg.norm(pp) < max_range and pp[1] > -1.0) else False for pp in xyz]
    return np.array(xyz)[xyz_mask]


def cluster_point_cloud(point_cloud):
    """Finds clusters in the given point cloud (x,y,z only)."""
    db_clusters = DBSCAN(eps=1.0, min_samples=10).fit(point_cloud)
    # Following works better in detecting pedestrians at a distance but I don't think it is very practical:
    # db_clusters = DBSCAN(eps=1.0, min_samples=3).fit(point_cloud)
    core_samples_mask = np.zeros_like(db_clusters.labels_, dtype=bool)
    core_samples_mask[db_clusters.core_sample_indices_] = True
    labels = db_clusters.labels_
    return labels


def filter_clustered_points(cluster_labels, labeled_point_cloud):
    """Removes the points that doesn't look like a car or pedestrian. (like buildings/road)"""
    clusters = []
    if len(cluster_labels) > 0 and len(labeled_point_cloud) > 0:
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        if n_clusters >= 1:
            for k in unique_labels:
                if k >= 0:
                    class_member_mask = (cluster_labels == k)
                    points_in_cluster = labeled_point_cloud[class_member_mask]
                    # Check x and y lengths and the volume to make sure it is at most at the size of a car.
                    max_x = max(points_in_cluster[:, 0])
                    min_x = min(points_in_cluster[:, 0])
                    max_y = max(points_in_cluster[:, 2])
                    min_y = min(points_in_cluster[:, 2])
                    if (len(points_in_cluster) > 2 and
                            (max_x - min_x) < 6 and
                            (max_y - min_y) < 6 and
                            (max_x - min_x)*(max_y - min_y) < 16):
                        clusters.append(LidarCluster(cluster_points=points_in_cluster[:],
                                                     min_x=min_x,
                                                     max_x=max_x,
                                                     min_y=min_y,
                                                     max_y=max_y))
    return clusters
