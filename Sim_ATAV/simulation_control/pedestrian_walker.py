"""Defines PedestrianWalker class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import math


class PedestrianWalker(object):
    """PedestrianWalker class provides necessary functionality to compute and update pedestrian motion."""
    def __init__(self, pedestrian, supervisor_control):
        self.BODY_PARTS_NUMBER = 13
        self.WALK_SEQUENCES_NUMBER = 8
        self.ROOT_HEIGHT = 1.27
        self.CYCLE_TO_DISTANCE_RATIO = 0.22
        self.current_height_offset = 0
        self.joints_position_field = []
        self.joint_names = [
            "leftArmAngle", "leftLowerArmAngle", "leftHandAngle",
            "rightArmAngle", "rightLowerArmAngle", "rightHandAngle",
            "leftLegAngle", "leftLowerLegAngle", "leftFootAngle",
            "rightLegAngle", "rightLowerLegAngle", "rightFootAngle",
            "headAngle"
        ]
        self.height_offsets = [  # those coefficients are empirical coefficients which result in a realistic walking gait
            -0.02, 0.04, 0.08, -0.03, -0.02, 0.04, 0.08, -0.03
        ]
        self.angles = [  # those coefficients are empirical coefficients which result in a realistic walking gait
            [-0.52, -0.15,  0.58,  0.7,   0.52,  0.17, -0.36, -0.74],  # left arm
            [ 0.0,  -0.16, -0.7,  -0.38, -0.47, -0.3,  -0.58, -0.21],  # left lower arm
            [ 0.12,  0.0,   0.12,  0.2,   0.0,  -0.17, -0.25,  0.0 ],  # left hand
            [ 0.52,  0.17, -0.36, -0.74, -0.52, -0.15,  0.58,  0.7 ],  # right arm
            [-0.47, -0.3,  -0.58, -0.21,  0.0,  -0.16, -0.7,  -0.38],  # right lower arm
            [ 0.0,  -0.17, -0.25,  0.0,   0.12,  0.0,   0.12,  0.2 ],  # right hand
            [-0.55, -0.85, -1.14, -0.7,  -0.56,  0.12,  0.24,  0.4 ],  # left leg
            [ 1.4,   1.58,  1.71,  0.49,  0.84,  0.0,   0.14,  0.26],  # left lower leg
            [ 0.07,  0.07, -0.07, -0.36,  0.0,   0.0,   0.32, -0.07],  # left foot
            [-0.56,  0.12,  0.24,  0.4,  -0.55, -0.85, -1.14, -0.7 ],  # right leg
            [ 0.84,  0.0,   0.14,  0.26,  1.4,   1.58,  1.71,  0.49],  # right lower leg
            [ 0.0,   0.0,   0.42, -0.07,  0.07,  0.07, -0.07, -0.36],  # right foot
            [ 0.18,  0.09,  0.0,   0.09,  0.18,  0.09,  0.0,   0.09]   # head
        ]
        self.waypoints = []
        self.waypoints_distance = []
        self.number_of_waypoints = 0
        self.pedestrian = pedestrian
        self.speed = self.pedestrian.target_speed
        self.distance = 0
        self.prev_time = 0
        self.set_trajectory()
        self.supervisor_control = supervisor_control
        self.abstraction_field =self.supervisor_control.get_obj_field(self.pedestrian, "abstractionLevel")
        if self.abstraction_field is not None:
            self.abstraction_level = self.abstraction_field.getSFInt32()
        else:
            self.abstraction_level = 1
        if self.abstraction_level > 0:
            for i in range(0, self.BODY_PARTS_NUMBER):
                self.joints_position_field.append(self.supervisor_control.get_obj_field(self.pedestrian, self.joint_names[i]))

    def set_trajectory(self):
        """Generates a list of waypoints from the given trajectory."""
        self.number_of_waypoints = int(len(self.pedestrian.trajectory)/2)
        for i in range(0, self.number_of_waypoints):
            self.waypoints.append([])
            self.waypoints[i].append(float(self.pedestrian.trajectory[i*2]))
            self.waypoints[i].append(float(self.pedestrian.trajectory[i*2 + 1]))
        # compute waypoints distance
        for i in range(0, self.number_of_waypoints):
            x = self.waypoints[i][0] - self.waypoints[(i + 1) % self.number_of_waypoints][0]
            z = self.waypoints[i][1] - self.waypoints[(i + 1) % self.number_of_waypoints][1]
            if i == 0:
                self.waypoints_distance.append(math.sqrt(x * x + z * z))
            else:
                self.waypoints_distance.append(self.waypoints_distance[i - 1] + math.sqrt(x * x + z * z))

    def compute_movement(self):
        """Computes the new position, rotation and joint angles based on the speed and trajectory."""
        time = self.supervisor_control.get_time()
        current_position = self.supervisor_control.get_obj_position_3D(self.pedestrian)

        new_joint_angles = []
        if self.abstraction_level > 0:
            current_sequence = int(((time * self.speed) / self.CYCLE_TO_DISTANCE_RATIO) % self.WALK_SEQUENCES_NUMBER)
            # compute the ratio 'distance already covered between way-point(X) and way-point(X+1)' / 'total distance between way-point(X) and way-point(X+1)'
            ratio = (time * self.speed) / self.CYCLE_TO_DISTANCE_RATIO - int(((time * self.speed) / self.CYCLE_TO_DISTANCE_RATIO))

            for i in range(0, self.BODY_PARTS_NUMBER):
                current_angle = self.angles[i][current_sequence] * (1 - ratio) + self.angles[i][(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio
                new_joint_angles.append(current_angle)

            # adjust height
            self.current_height_offset = self.height_offsets[current_sequence] * (1 - ratio) + self.height_offsets[(current_sequence + 1) % self.WALK_SEQUENCES_NUMBER] * ratio

        # move everything
        time_delta = time - self.prev_time
        self.distance += time_delta * self.speed
        #distance = time * self.speed
        if self.number_of_waypoints > 1:
            relative_distance = self.distance - int(self.distance / self.waypoints_distance[self.number_of_waypoints - 1]) * self.waypoints_distance[self.number_of_waypoints - 1]

            for i in range(0, self.number_of_waypoints):
                if self.waypoints_distance[i] > relative_distance:
                    break

            distance_ratio = 0
            if i == 0:
                distance_ratio = relative_distance / self.waypoints_distance[0]
            else:
                distance_ratio = (relative_distance - self.waypoints_distance[i - 1]) / (self.waypoints_distance[i] - self.waypoints_distance[i - 1])
            x = distance_ratio * self.waypoints[(i + 1) % self.number_of_waypoints][0] + (1 - distance_ratio) * self.waypoints[i][0]
            z = distance_ratio * self.waypoints[(i + 1) % self.number_of_waypoints][1] + (1 - distance_ratio) * self.waypoints[i][1]
            root_3D_position = [x, self.ROOT_HEIGHT + self.current_height_offset, z]
            angle = math.atan2(self.waypoints[(i + 1) % self.number_of_waypoints][0] - self.waypoints[i][0], 
                               self.waypoints[(i + 1) % self.number_of_waypoints][1] - self.waypoints[i][1])
            rotation = [0, 1, 0, angle]
            self.prev_time = time
        else:
            root_3D_position = current_position
            rotation = self.supervisor_control.get_obj_rotation(self.pedestrian)
        return (root_3D_position, rotation, new_joint_angles)

    def apply_movement(self, root_3D_position, rotation, new_joint_angles):
        """Realize the computed movement. (New position, rotation and joint angles)"""
        if self.abstraction_level > 0:
            for i in range(0, self.BODY_PARTS_NUMBER):
                current_angle = new_joint_angles[i]
                self.joints_position_field[i].setSFFloat(current_angle)
        self.supervisor_control.set_obj_position_3D(self.pedestrian, root_3D_position)
        self.supervisor_control.set_obj_rotation(self.pedestrian, rotation)

    def move(self):
        """Move the pedestrian: Computes and applies the movement."""
        (root_3D_position, rotation, new_joint_angles) = self.compute_movement()
        self.apply_movement(root_3D_position, rotation, new_joint_angles)
