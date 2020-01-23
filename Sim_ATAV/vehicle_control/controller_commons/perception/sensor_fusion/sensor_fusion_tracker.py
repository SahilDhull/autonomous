"""Defines SensorFusionTracker class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import math
import numpy as np
from pykalman import AdditiveUnscentedKalmanFilter, UnscentedKalmanFilter


class SensorFusionTracker(object):
    """A simple Sensor Fusion implementation which utilizes an Unscented Kalman Filter."""
    SENSOR_TYPE_RADAR = 'radar'
    SENSOR_TYPE_CAMERA = 'camera'
    SENSOR_TYPE_GPS = 'gps'
    SENSOR_TYPE_COMPASS = 'compass'
    SENSOR_TYPE_RADAR_CAMERA = 'radar_camera'
    SENSOR_TYPE_GPS_COMPASS = 'gps_compass'

    def __init__(self, time_step=0.01, initial_state_mean=[0, 0, 0, 0, 0], filter_type='akf', object_type='car'):
        self.delta_t = time_step
        self.transition_covariance = np.eye(5)
        self.random_state = np.random.RandomState(0)
        self.observation_covariance = np.eye(5) + self.random_state.randn(5, 5) * 0.1
        self.tracked_object_state_cov = np.eye(5) * 0.9 + 0.1 * np.ones([5, 5])
        self.tracked_object_state = initial_state_mean[:]
        self.object_type = object_type

        if filter_type == 'akf':
            if object_type == 'car':
                self.transition_function = self.additive_ctrv_model
            else:
                self.transition_function = self.additive_linear_motion_model
            self.filter = AdditiveUnscentedKalmanFilter(initial_state_mean=self.tracked_object_state,
                                                        initial_state_covariance=self.tracked_object_state_cov)
        else:
            if object_type == 'car':
                self.transition_function = self.ctrv_model
            else:
                self.transition_function = self.linear_motion_model
            self.filter = UnscentedKalmanFilter(initial_state_mean=self.tracked_object_state,
                                                initial_state_covariance=self.tracked_object_state_cov,
                                                random_state=self.random_state)
        self.sensor_function_dict = {self.SENSOR_TYPE_RADAR: self.sensor_function_radar,
                                     self.SENSOR_TYPE_CAMERA: self.sensor_function_camera,
                                     self.SENSOR_TYPE_GPS: self.sensor_function_gps,
                                     self.SENSOR_TYPE_COMPASS: self.sensor_function_compass,
                                     self.SENSOR_TYPE_RADAR_CAMERA: self.sensor_function_radar_camera,
                                     self.SENSOR_TYPE_GPS_COMPASS: self.sensor_function_gps_compass}

    def get_projection(self, time_step=None):
        """Get a projection based on the transition function without using any measurements."""
        return self.transition_function(self.tracked_object_state, time_step=time_step)

    def get_projection_with_state(self, init_state, time_step=0.01, time_duration=None):
        """Get a projection based on the transition function without using any measurements."""
        if time_duration > time_step:
            state = init_state[:]
            num_steps = int(time_duration / time_step)
            for _step in range(num_steps):
                state = self.transition_function(state, time_step=time_step)
        else:
            state = self.transition_function(init_state, time_step=time_duration)
        return state[:]

    def set_object_type(self, object_type):
        self.object_type = object_type
        if object_type == 'car':
            self.transition_function = self.additive_ctrv_model
        else:
            self.transition_function = self.additive_linear_motion_model

    def ctrv_model(self, state, noise=np.array([0, 0, 0, 0, 0]), time_step=None):
        """Continuous Turn Rate and Velocity model.
        States: [p_x, p_y, speed, yaw, yaw_rate]"""
        if time_step is None:
            delta_t = self.delta_t
        else:
            delta_t = time_step

        return ctrv_model(state=state, noise=noise, delta_t=delta_t)

    def additive_ctrv_model(self, state, time_step=None):
        """CTRV model to use with AdditiveUnscentedKalmanFilter.
        Noise for the transition function is set to zero.
        Noise is somehow handled in the filter itself."""
        return self.ctrv_model(state, np.array([0, 0, 0, 0, 0]), time_step=time_step)

    def linear_motion_model(self, state, noise=np.array([0, 0, 0, 0, 0]), time_step=None):
        """Linear motion with constant velocity.
        States: [p_x, p_y, speed, yaw, yaw_rate]"""
        if time_step is None:
            delta_t = self.delta_t
        else:
            delta_t = time_step

        p_x = state[0]
        p_y = state[1]
        spd = state[2]
        yaw = state[3]
        yaw_rate = state[4]
        px_new = p_x - spd * np.sin(yaw) * delta_t + noise[0]  # +ve x is towards left. But the yaw angle is clockwise.
        py_new = p_y + spd * np.cos(yaw) * delta_t + noise[1]
        spd_new = spd + noise[2]
        yaw_new = yaw + noise[3]
        yaw_rate_new = yaw_rate + noise[4]
        return np.array([px_new, py_new, spd_new, yaw_new, yaw_rate_new])

    def additive_linear_motion_model(self, state, time_step=None):
        """Linear model to use with AdditiveUnscentedKalmanFilter.
        Noise for the transition function is set to zero.
        Noise is somehow handled in the filter itself."""
        return self.linear_motion_model(state, np.array([0, 0, 0, 0, 0]), time_step=time_step)

    def sensor_function_radar(self, state, noise=np.array([0, 0, 0])):
        """Sensor function for radar. Describes how states relate to the radar sensor data.
        Expected sensor data format: [angle, distance, speed]"""
        obs_f = np.array([-math.atan2(state[0], state[1]), math.sqrt(state[0]**2 + state[1]**2), state[2]])
        return obs_f + noise

    def sensor_function_gps(self, state, noise=np.array([0, 0, 0])):
        """Sensor function for GPS. Describes how states relate to the sensor data.
        Expected sensor data format: [position x, position y, speed]"""
        obs_f = np.array([state[0], state[1], state[2]])
        return obs_f + noise

    def sensor_function_compass(self, state, noise=np.array([0])):
        """Sensor function for compass. Describes how states relate to the sensor data.
        Expected sensor data format: [yaw angle]"""
        obs_f = np.array([state[3]])
        return obs_f + noise

    def sensor_function_gps_compass(self, state, noise=np.array([0, 0, 0, 0])):
        """Sensor function for GPS and Compass together. Describes how states relate to the sensor data.
        Expected sensor data format: first gps data, then compass: [position x, position y, speed, yaw_angle]"""
        obs_f = self.sensor_function_gps(state, noise[0:3])
        obs_f = np.append(obs_f, self.sensor_function_compass(state, noise[3]))
        return obs_f

    def sensor_function_camera(self, state, noise=np.array([0, 0, 0, 0])):
        """Sensor function for camera detections.
        Describes how states relate to the camera object detection data.
        Expected object detection data format for detection box (everything in pixels):
        [center_x, center_y, width, height]"""
        pass

    def sensor_function_radar_camera(self, state, noise=np.array([0, 0, 0, 0, 0, 0, 0])):
        """Sensor function for radar camera detections together.
        Describes how states relate to the sensor data.
        Expected first radar and then camera data.:
        [angle, distance, speed, center_x, center_y, width, height]"""
        pass

    def get_estimates(self, measurements, sensor_type, mask_measurement=False):
        """Utilizes the Kalman Filter to get new state estimates using the current observations."""
        observation_covariance = np.eye(len(measurements)) + self.random_state.randn(len(measurements), len(measurements)) * 0.1
        if mask_measurement:
            measurements = None
        self.tracked_object_state, self.tracked_object_state_cov = \
            self.filter.filter_update(filtered_state_mean=self.tracked_object_state,
                                      filtered_state_covariance=self.tracked_object_state_cov,
                                      observation=measurements,
                                      transition_function=self.transition_function,
                                      transition_covariance=self.transition_covariance,
                                      observation_function=self.sensor_function_dict[sensor_type],
                                      observation_covariance=observation_covariance)
        return self.tracked_object_state, self.tracked_object_state_cov


def ctrv_model(state, noise=np.array([0, 0, 0, 0, 0]), delta_t=0.0):
    """Continuous Turn Rate and Velocity model.
    States: [p_x, p_y, speed, yaw, yaw_rate]"""
    p_x = state[0]
    p_y = state[1]
    spd = state[2]
    yaw = state[3]
    yaw_rate = state[4]

    if abs(yaw_rate) < 0.0001:
        px_new = p_x - spd * np.sin(yaw) * delta_t + noise[0]  # +ve x is towards left. But the yaw angle is clockwise.
        py_new = p_y + spd * np.cos(yaw) * delta_t + noise[1]
    else:
        px_new = p_x + (spd / yaw_rate) * (np.cos(yaw + yaw_rate*delta_t) - np.cos(yaw)) + noise[0]
        py_new = p_y + (spd / yaw_rate) * (np.sin(yaw + yaw_rate*delta_t) - np.sin(yaw)) + noise[1]
    spd_new = spd + noise[2]
    yaw_new = yaw + yaw_rate * delta_t + noise[3]
    yaw_rate_new = yaw_rate + noise[4]
    # TODO: Go to https://wauner.github.io/projects/unscented/ctrv/ for better noise modeling.
    return np.array([px_new, py_new, spd_new, yaw_new, yaw_rate_new])