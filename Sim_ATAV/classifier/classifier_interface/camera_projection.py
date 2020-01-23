"""Defines CameraProjection class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


import math
import numpy as np


class CameraProjection(object):
    """CameraProjection class handles the computations for 
    projection of an object in the world to the camera coordinates."""

    def __init__(self, width=None, height=None, horizontal_FOV=None):
        self.cam_width = width
        self.cam_height = height
        self.cam_hor_fov = horizontal_FOV
        # self.cam_vert_fov = 2 * math.atan(math.tan(horizontal_FOV * 0.5) / (width / height))
        if height is None or width is None or horizontal_FOV is None:
            self.projection_matrix = np.zeros([3,4])
        else:
            self.set_camera_specific_settings(width, height, horizontal_FOV)

    def set_camera_specific_settings(self, width, height, horizontal_FOV):
        """Set camera parameters for building projection matrix."""
        # vertical_FOV computation is a bit different with the following: Try and see which works best.
        # vertical_FOV = (horizontal_FOV*height/width)
        self.cam_width = width
        self.cam_height = height
        self.cam_hor_fov = horizontal_FOV
        vertical_FOV = 2 * math.atan(math.tan(horizontal_FOV * 0.5) / (width / height))
        # self.cam_vert_fov = vertical_FOV
        f_y = height*0.5/math.tan(0.5*vertical_FOV)
        f_x = width*0.5/math.tan(0.5*horizontal_FOV)
        # These were the values I manually computed. They work well, can be used as a reference.
        #f_x = 1506.5
        #f_y = 1581.3
        center_x = width/2.0
        center_y = height/2.0
        self.projection_matrix[0, 0] = -f_x
        self.projection_matrix[1, 1] = -f_y
        self.projection_matrix[0, 2] = center_x
        self.projection_matrix[1, 2] = center_y
        self.projection_matrix[2, 2] = 1

    def convert_from_local_to_world_coordinates(self,
                                                object_rotation_matrix_3d,
                                                object_position,
                                                local_coordinates):
        """Take the object position, local coordinates of a point in the
        object's coordinate system and the rotation matrix of the object.
        Return the position of the point in the world coordinate system."""
        return np.matmul(object_rotation_matrix_3d, local_coordinates) + object_position

    def convert_from_world_to_camera_coordinates(self,
                                                 camera_rotation_matrix_3d,
                                                 camera_position,
                                                 point_world_coordinates):
        """Consider the position and rotation of the camera,
        and convert the world coordinates of the point to the camera coordinate system,
        which is defined wrt the camera position and rotation."""
        R = np.append(np.transpose(camera_rotation_matrix_3d), np.zeros([3,1]), axis=1)
        R = np.append(R, np.zeros([1,4]), axis=0)
        R[3, 3] = 1.0
        C = np.eye(4)
        C[0, 3] = -camera_position[0]
        C[1, 3] = -camera_position[1]
        C[2, 3] = -camera_position[2]
        W = np.append(point_world_coordinates, np.ones([1,1]), axis=0)
        return np.matmul(R, np.matmul(C, W))

    def convert_from_camera_to_image_coordinates(self,
                                                 camera_coordinates_of_point):
        return np.matmul(self.projection_matrix, camera_coordinates_of_point)
