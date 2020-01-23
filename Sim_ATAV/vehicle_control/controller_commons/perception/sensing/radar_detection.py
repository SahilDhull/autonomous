"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation.radar_object import RadarObject
from Sim_ATAV.vehicle_control.controller_commons import controller_commons


class RadarDetection(object):
    def __init__(self, radar_device=None,radar_relative_pos=(0.0, 0.0)):
        self.radar = radar_device
        self.radar_relative_position = [radar_relative_pos[0], radar_relative_pos[1]]

    def read_radar_and_find_objects(self):
        """Reads radar targets and creates a list of detected objects."""
        radar_detected_objects = []
        radar_targets = []
        is_read = False
        if self.radar is not None:
            is_read = True
            radar_targets = self.radar.getTargets()
            accepted_hor_fov = 0.95 * self.radar.getHorizontalFov() / 2.0
            # Reason for *0.95: There is some distortion at the ends of FOV (because of the vehicle sizes).
            for radar_target in radar_targets:
                if (self.radar.getMinRange() < radar_target.distance < self.radar.getMaxRange()
                        and -accepted_hor_fov < radar_target.azimuth < accepted_hor_fov):
                    position = controller_commons.polar_coordinates_to_cartesian(radar_target.distance,
                                                                                 radar_target.azimuth)
                    position = [position[0] + self.radar_relative_position[0],
                                position[1] + self.radar_relative_position[1]]
                    radar_detected_objects.append(RadarObject(radar_target=radar_target,
                                                              object_class=self.radar_object_class(radar_target),
                                                              relative_position=position))
        return radar_detected_objects, radar_targets, is_read

    def radar_object_class(self, radar_target):
        """Estimate object type from radar signal power."""
        power = pow(10, radar_target.received_power / 10.0) * 0.001 * pow(radar_target.distance, 4.0)
        if power < 0.005:  # Motorcycle or pedestrian
            object_class = RadarObject.OBJECT_PEDESTRIAN
        elif power > 0.015:  # Large vehicle (bus or truck)
            object_class = RadarObject.OBJECT_TRUCK
        else:  # Car
            object_class = RadarObject.OBJECT_CAR
        return object_class
