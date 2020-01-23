"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from Sim_ATAV.vehicle_control.controller_commons import controller_commons
from Sim_ATAV.vehicle_control.controller_commons.perception.sensor_fusion.sensor_fusion_tracker import SensorFusionTracker


class EgoStateSensorFusion(object):
    def __init__(self):
        self.state = []
        self.gps = None
        self.compass = None
        self.accelerometer = None
        self.self_sensor_fusion_tracker = None
        self.self_current_state = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.sensor_gps_position_m = [0.0, 0.0, 0.0]
        self.sensor_gps_speed_m_s = 0.0
        self.sensor_compass_bearing_rad = 0.0
        self.prev_sensor_compass_bearing_rad = None
        self.prev_time_ms = None
        self.yaw_rate = 0.0

    def get_speed_ms(self):
        return self.self_current_state[2]

    def get_position(self):
        return self.self_current_state[0:2]

    def get_yaw_angle(self):
        return self.self_current_state[3]

    def get_yaw_rate(self):
        return self.self_current_state[4]

    def set_gps_sensor(self, gps_device):
        self.gps = gps_device

    def set_compass_sensor(self, compass_device):
        self.compass = compass_device

    def set_accelerometer_sensor(self, accelerometer_device):
        self.accelerometer = accelerometer_device

    def update_states(self, cur_time_ms):
        self.read_sensors(cur_time_ms)
        self.do_sensor_fusion()

    def read_sensors(self, cur_time_ms):
        # -------------- Read Sensors for ego states ----------------
        (self.sensor_gps_position_m, self.sensor_gps_speed_m_s) = controller_commons.read_gps_sensor(self.gps)
        self.sensor_compass_bearing_rad = controller_commons.read_compass_sensor(self.compass)
        if self.prev_sensor_compass_bearing_rad is not None:
            self.yaw_rate = ((self.sensor_compass_bearing_rad - self.prev_sensor_compass_bearing_rad) /
                             ((cur_time_ms - self.prev_time_ms) / 1000.0))
        else:
            self.yaw_rate = 0.0
        self.prev_time_ms = cur_time_ms
        self.prev_sensor_compass_bearing_rad = self.sensor_compass_bearing_rad

    def do_sensor_fusion(self):
        # -------------- Sensor Fusion ----------------
        # ************ Sensor Fusion for own states (GPS + Compass) ************
        if self.self_sensor_fusion_tracker is None:

            self.self_current_state = [self.sensor_gps_position_m[0],
                                       self.sensor_gps_position_m[2],
                                       self.sensor_gps_speed_m_s,
                                       self.sensor_compass_bearing_rad,
                                       self.yaw_rate]
            if self.sensor_gps_speed_m_s > 50.0 or self.sensor_gps_speed_m_s < -20.0:
                # Filter out errors in gps speed reading
                self.sensor_gps_speed_m_s = 0.0
                self.self_current_state[2] = self.sensor_gps_speed_m_s
            # Initiate self sensor fusion tracker
            self.self_sensor_fusion_tracker = SensorFusionTracker(initial_state_mean=self.self_current_state,
                                                                  filter_type='ukf')
        else:
            if self.gps is not None and self.compass is not None:
                measurement = [self.sensor_gps_position_m[0],
                               self.sensor_gps_position_m[2],
                               self.sensor_gps_speed_m_s,
                               self.sensor_compass_bearing_rad]
                (self.self_current_state, state_cov) = self.self_sensor_fusion_tracker.get_estimates(
                    measurements=measurement,
                    sensor_type=SensorFusionTracker.SENSOR_TYPE_GPS_COMPASS)
            elif self.gps is not None:
                measurement = [self.sensor_gps_position_m[0], self.sensor_gps_position_m[2], self.sensor_gps_speed_m_s]
                (self.self_current_state, state_cov) = \
                    self.self_sensor_fusion_tracker.get_estimates(measurements=measurement,
                                                                  sensor_type=SensorFusionTracker.SENSOR_TYPE_GPS)
            elif self.compass is not None:
                measurement = [self.sensor_compass_bearing_rad]
                (self.self_current_state, state_cov) = \
                    self.self_sensor_fusion_tracker.get_estimates(measurements=measurement,
                                                                  sensor_type=SensorFusionTracker.SENSOR_TYPE_COMPASS)
            else:
                self.self_current_state = [0.0, 0.0, 0.0, 0.0, 0.0]
