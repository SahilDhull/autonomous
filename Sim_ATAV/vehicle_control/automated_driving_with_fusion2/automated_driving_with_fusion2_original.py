"""Defines SimpleSensorFusionControl class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import math
import pickle
import numpy as np
from Sim_ATAV.common.controller_communication_interface import ControllerCommunicationInterface
from Sim_ATAV.vehicle_control.base_controller.base_controller import BaseCarController
from Sim_ATAV.vehicle_control.controller_commons import controller_commons
from Sim_ATAV.vehicle_control.controller_commons.visualization.camera_info_display import CameraInfoDisplay
from Sim_ATAV.vehicle_control.generic_stanley_controller.generic_stanley_controller \
    import GenericStanleyController
from Sim_ATAV.vehicle_control.generic_pid_controller.generic_pid_controller import GenericPIDController
from Sim_ATAV.classifier.classifier_interface.classifier import Classifier
from Sim_ATAV.classifier.classifier_interface.ground_truth_generator import GroundTruthGenerator
from Sim_ATAV.vehicle_control.controller_commons.perception.sensing.camera_detection import CameraDetection
from Sim_ATAV.vehicle_control.controller_commons.perception.sensing.radar_detection import RadarDetection
from Sim_ATAV.vehicle_control.controller_commons.perception.sensing.lidar_detection import LidarDetection
from Sim_ATAV.vehicle_control.controller_commons.perception.sensor_fusion.simple_sensor_fusion import SimpleSensorFusion
from Sim_ATAV.vehicle_control.controller_commons.perception.sensing.radio_detection import RadioDetection
from Sim_ATAV.vehicle_control.controller_commons.perception.sensor_fusion.ego_state_sensor_fusion \
    import EgoStateSensorFusion
from Sim_ATAV.vehicle_control.controller_commons.visualization.sensor_visualization import SensorVisualization
from Sim_ATAV.vehicle_control.controller_commons.visualization.console_output import ConsoleOutput
from Sim_ATAV.vehicle_control.controller_commons.planning.path_planner import PathPlanner
from .high_level_control import HighLevelControl
from .low_level_control import LowLevelControl
from .communication_module import CommunicationModule

# from vel_acc_to_throttle import *

WORLD_TIME_STEP_MS = 10
HAS_DEBUG_DISPLAY = True
SENSOR_TYPE = 'Actual'  # 'Actual' #'Perfect'
DEBUG_MODE = False

# target_a = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
# target_v = [5.0, 7.0710678118654755, 8.660254037844387, 10.0, 11.180339887498949, 12.24744871391589, 13.228756555322953, 14.142135623730951, 15.000000000000002, 15.811388300841898, 16.583123951777, 17.320508075688775, 18.02775637731995, 18.70828693386971, 19.364916731037088, 20.000000000000004, 20.615528128088304, 21.213203435596427, 21.79449471770337, 22.360679774997898, 22.912878474779202, 23.45207879911715, 23.9791576165636, 24.494897427831784, 25.000000000000004, 25.495097567963928, 25.980762113533164, 26.45751311064591, 26.925824035672523]
target_throttle = [0.15, 0.25, 0.3, 0.3, 0.3, 0.3, 0.35, 0.1, 0.1, 0.25, 0.25, 0.35, 0.25, 0.15, 0.15, 0.15, 0.2, 0.2, 0.3, 0.3, 0.3, 0.25, 0.25, 0.25, 0.35, 0.25, 0.25, 0.25, 0.25, 0.35, 0.25, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.45, 0.55, 0.55, 0.6, 0.6, 0.6, 0.65, 0.8, 0.85, 0.7, 0.7, 0.75, 0.75, 0.75, 0.8, 0.8, 0.8, 0.85, 0.85, 0.85, 0.85, 0.9, 0.9, 0.95, 0.95, 0.95, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
target_t = [0.0, 2.23, 3.16, 3.87, 4.47, 5.0, 5.47, 5.93, 6.39, 6.84, 7.27, 7.68, 8.08, 8.47, 8.86, 9.25, 9.63, 10.01, 10.38, 10.74, 11.09, 11.43, 11.77, 12.11, 12.44, 12.76, 13.08, 13.4, 13.72, 14.03, 14.33, 14.63, 14.93, 15.23, 15.53, 15.82, 16.11, 16.39, 16.67, 16.94, 17.21, 17.47, 17.72, 17.97, 18.21, 18.45, 18.68, 18.91, 19.14, 19.37, 19.59, 19.81, 20.02, 20.23, 20.44, 20.65, 20.86, 21.06, 21.26, 21.46, 21.65, 21.84, 22.03, 22.22, 22.41, 22.6, 22.78, 22.96, 23.14, 23.32, 23.49, 23.66, 23.83, 24.0, 24.17, 24.34, 24.51, 24.68, 24.85, 25.01, 25.17, 25.33, 25.49, 25.65, 25.8, 25.95, 26.1, 26.25, 26.4, 26.55, 26.7, 26.85, 27.0, 27.15, 27.3, 27.45, 27.6, 27.75, 27.89, 28.03, 28.17, 28.31, 28.45, 28.59, 28.73, 28.87, 29.0, 29.13, 29.26, 29.39, 29.52, 29.65, 29.78, 29.91, 30.04, 30.17, 30.3, 30.43, 30.56, 30.69, 30.82, 30.95, 31.08, 31.21, 31.34, 31.47, 31.59, 31.71, 31.83, 31.95, 32.07, 32.19, 32.31, 32.43, 32.55, 32.67, 32.79, 32.91, 33.03, 33.14, 33.25, 33.36, 33.47, 33.58, 33.69, 33.8, 33.91, 34.02, 34.13, 34.24, 34.35, 34.46, 34.57, 34.68, 34.79, 34.9, 35.01, 35.12, 35.23, 35.34, 35.45, 35.56, 35.67, 35.78, 35.89, 36.0, 36.11, 36.22, 36.33, 36.44, 36.55, 36.66, 36.76, 36.86, 36.96, 37.06, 37.16, 37.26, 37.36, 37.46, 37.56, 37.66, 37.76, 37.86, 37.96, 38.06, 38.16, 38.26, 38.36, 38.46, 38.56, 38.66, 38.76, 38.86, 38.96, 39.05, 39.14, 39.23, 39.32, 39.41, 39.5, 39.59, 39.68, 39.77, 39.86, 39.95, 40.04, 40.13, 40.22, 40.31, 40.4, 40.49, 40.58, 40.67, 40.76, 40.85, 40.94, 41.03, 41.12, 41.21, 41.3, 41.39, 41.48, 41.57, 41.66, 41.75, 41.84, 41.93, 42.02, 42.11, 42.2, 42.29, 42.38, 42.47, 42.56, 42.65, 42.74, 42.83, 42.92, 43.01, 43.1, 43.19, 43.28, 43.37, 43.46, 43.55, 43.64, 43.73, 43.82, 43.91, 44.0, 44.09, 44.18, 44.27, 44.36, 44.45, 44.53, 44.61, 44.69, 44.77, 44.85, 44.93, 45.01, 45.09, 45.17, 45.25, 45.33, 45.41, 45.49, 45.57, 45.65, 45.73, 45.81, 45.89, 45.97, 46.05, 46.13, 46.21, 46.29, 46.37, 46.45, 46.53, 46.61, 46.69, 46.77, 46.85, 46.93, 47.01, 47.09, 47.17, 47.25, 47.33, 47.41]

time_index = 0
img_cnt = 1
data_dict = {}



class AutomatedDrivingControlWithFusion2(BaseCarController):
    """AutomatedDrivingControlWithFusion class is a car controller class for Webots.
    It is used for experimenting basic automated driving capabilities for
    perception system test purposes."""

    CAMERA_TO_FRONT_DISTANCE = 2.3  # 2.3 m is the distance from Prius top sensor location to the very front of the car
    LIDAR_TO_FRONT_DISTANCE = 2.3
    CAMERA_MAIN_RELATIVE_POSITION = [0.0, 1.3]
    LIDAR_MAIN_RELATIVE_POSITION = [0.0, 1.3]
    RADAR_FRONT_RELATIVE_POSITION = [0.0, 3.6]
    FRONT_TO_REAR_WHEELS_DISTANCE = 3.6  # Approximate (this is intentionally longer than the actual wheel base
    # for smoother operation)

    CAMERA_LOCAL_COORDINATES = [0.0, 1.3, 1.1]
    CAMERA_X_ROT_ANGLE = -0.01
    CAMERA_LOCAL_ROTATION = np.array([[1.0, 0.0, 0.0],
                                      [0.0, math.cos(CAMERA_X_ROT_ANGLE), -math.sin(CAMERA_X_ROT_ANGLE)],
                                      [0.0, math.sin(CAMERA_X_ROT_ANGLE), math.cos(CAMERA_X_ROT_ANGLE)]])

    CAR_FRONT_TRIANGLE_LINE1_M = -192/126  # old value: -0.6  # Line 1 m for front triangle.
    CAR_FRONT_TRIANGLE_LINE1_B = 1142.9  # Old value: 526  # Line 1 b for front triangle.
    CAR_FRONT_TRIANGLE_LINE2_M = 192/126  # old value: 0.6  # Line 2 m for front triangle.
    CAR_FRONT_TRIANGLE_LINE2_B = -758.9  # Old value: -202  # Line 2 b for front triangle.

    PED_FRONT_TRIANGLE_LINE1_M = -192/204  # old value: -0.6  # Line 1 m for front triangle.
    PED_FRONT_TRIANGLE_LINE1_B = 779.3  # Old value: 526  # Line 1 b for front triangle.
    PED_FRONT_TRIANGLE_LINE2_M = 192/204  # old value: 0.6  # Line 2 m for front triangle.
    PED_FRONT_TRIANGLE_LINE2_B = -395.3  # Old value: -202  # Line 2 b for front triangle.

    CLASSIFIER_PERIOD_MS = 100
    LIDAR_PERIOD_MS = 200
    RADAR_PERIOD_MS = 200
    MIN_EMERGENCY_BRAKE_DURATION_MS = 100.0
    MEASURE_EXEC_TIME = False
    LANE_WIDTH = 3.5
    MIN_STEERING_MANEUVER_MS = 2000.0
    EMERGENCY_STEERING_TTC = 1.0

    OBJECT_TRACKER_MAX_DISTANCE = 70.0

    def __init__(self, controller_parameters):
        (car_model, target_speed_kmh, target_lat_pos, self_vhc_id, slow_at_intersection, has_gpu, processor_id) = \
            controller_parameters
        BaseCarController.__init__(self, car_model)
        self.console_output = ConsoleOutput(DEBUG_MODE)
        self.slow_at_intersection = slow_at_intersection in ('True', 'true', 'yes', 'Yes')
        self.has_gpu = has_gpu in ('True', 'true', 'yes', 'Yes')
        self.processor_id = processor_id
        self.camera_device_name = 'camera'
        self.camera = None
        self.compass_device_name = 'compass'
        self.compass = None
        self.display_device_name = 'display'
        self.display = None
        self.camera_info_display = None
        self.sensor_display_device_name = 'sensor_display'
        self.sensor_display = None
        self.sensor_info_display = None
        self.gps_device_name = 'gps'
        self.gps = None
        self.receiver_device_name = 'receiver'
        self.receiver = None
        self.emitter_device_name = 'emitter'
        self.emitter = None
        self.lidar_main_device_name = 'velodyne'  # ibeo'  # 'velodyne'
        self.lidar_main = None
        self.radar_front_device_name = 'radar'
        self.radar_front = None
        self.target_speed_m_s = controller_commons.kmh_to_ms(float(target_speed_kmh))
        self.classifier = None
        self.classification_client = None
        self.obj_tracker = None
        self.ground_truth_generator = None
        self.controller_comm_interface = ControllerCommunicationInterface()
        self.target_lat_pos = float(target_lat_pos)
        self.target_bearing = 0.0
        self.lateral_controller = GenericStanleyController()
        self.lateral_controller.k = 0.5
        self.lateral_controller.k2 = 0.4
        self.lateral_controller.set_output_range(-0.8, 0.8)
        self.longitudinal_controller = GenericPIDController(0.15, 0.01, 0.0)
        self.longitudinal_controller.set_integrator_value_range(-20.0, 20.0)
        self.self_vhc_id = int(self_vhc_id)
        self.self_sensor_fusion_tracker = None
        self.last_segment_ind = 0
        self.self_current_state = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.last_segment_ind = 0
        self.very_risky_object_list = []
        self.risky_object_list = []
        self.proceed_w_caution_object_list = []
        self.detour_start_time = None
        self.camera_sensor = None
        self.radar_sensor = None
        self.lidar_sensor = None
        self.sensor_visualizer = None
        self.ground_truth_detector = RadioDetection(controller_communication_interface=self.controller_comm_interface,
                                                    ego_vhc_id=self.self_vhc_id)
        self.ego_state = EgoStateSensorFusion()
        self.perception_system = SimpleSensorFusion(self.ego_state)
        self.path_planner = PathPlanner()
        self.low_level_controller = LowLevelControl(ego_state=self.ego_state,
                                                    longitudinal_controller=self.longitudinal_controller,
                                                    lateral_controller=self.lateral_controller,
                                                    path_planner=self.path_planner)
        self.high_level_controller = HighLevelControl(ego_state=self.ego_state,
                                                      low_level_controller=self.low_level_controller,
                                                      path_planner=self.path_planner,
                                                      console_output=self.console_output)
        self.low_level_controller.set_parameter('long_position_offset', self.FRONT_TO_REAR_WHEELS_DISTANCE)
        self.high_level_controller.set_parameter('slow_down_at_intersections', True)
        # self.high_level_controller.set_parameter('target_speed_m_s', 50.0)
        self.radio_comm_module = None
        self.detection_perf_monitor = None
        self.visibility_monitor = None
        print('AutomatedDrivingControl Initialized: {}, {}'.format(car_model, self.target_speed_m_s))

    def start_devices(self):
        """Start the devices on the car and initialize objects like classifier."""
        # Start camera and the sensors:
        self.camera = self.getCamera(self.camera_device_name)
        if self.camera is not None:
            # self.camera.enable(self.CLASSIFIER_PERIOD_MS)
            # self.camera.enable(32)
            # self.camera.zoom = 1

            
            # self.camera.recognitionEnable(self.CLASSIFIER_PERIOD_MS)
            # self.camera.setFov(0.785398)
            # self.camera.setFocalDistance(30.0)
            # print("---")
            self.camera.enable(self.CLASSIFIER_PERIOD_MS)
            # self.classifier = Classifier(is_show_image=False, is_gpu=self.has_gpu, processor_id=self.processor_id)
            # self.classifier.start_classification_engine()
            # # self.obj_tracker = ObjectTracker()
            # self.ground_truth_generator = GroundTruthGenerator()
            # self.ground_truth_generator.set_camera_parameters(self.camera.getWidth(),
            #                                                   self.camera.getHeight(),
            #                                                   self.camera.getFov())
            # self.camera_sensor = CameraDetection(camera_device=self.camera,
            #                                      classifier=self.classifier,
            #                                      cam_relative_pos=(self.CAMERA_MAIN_RELATIVE_POSITION[0],
            #                                                        self.CAMERA_MAIN_RELATIVE_POSITION[1]))
            # self.perception_system.register_sensor(sensor_detector=self.camera_sensor,
            #                                        sensor_period=self.CLASSIFIER_PERIOD_MS)
        # self.display = self.getDisplay(self.display_device_name)
        self.camera_info_display = CameraInfoDisplay(self.display)
        if self.display is not None:
            if self.camera is not None:
                self.camera_info_display.attach_camera(self.camera)
        self.gps = self.getGPS(self.gps_device_name)
        if self.gps is not None:
            self.gps.enable(WORLD_TIME_STEP_MS)
            self.ego_state.set_gps_sensor(self.gps)
        self.compass = self.getCompass(self.compass_device_name)
        if self.compass is not None:
            self.compass.enable(WORLD_TIME_STEP_MS)
            self.ego_state.set_compass_sensor(self.compass)
        self.receiver = self.getReceiver(self.receiver_device_name)
        if self.receiver is not None:
            self.receiver.enable(WORLD_TIME_STEP_MS)
        self.emitter = self.getEmitter(self.emitter_device_name)
        self.lidar_main = self.getLidar(self.lidar_main_device_name)
        if self.lidar_main is not None:
            self.lidar_main.enable(self.LIDAR_PERIOD_MS)
            self.lidar_main.enablePointCloud()
            self.lidar_sensor = LidarDetection(lidar_device=self.lidar_main,
                                               lidar_relative_pos=(self.LIDAR_MAIN_RELATIVE_POSITION[0],
                                                                   self.LIDAR_MAIN_RELATIVE_POSITION[1]),
                                               lidar_layers=[7, 8, 9, 10])
            self.perception_system.register_sensor(sensor_detector=self.lidar_sensor,
                                                   sensor_period=self.LIDAR_PERIOD_MS)
        self.radar_front = self.getRadar(self.radar_front_device_name)
        if self.radar_front is not None:
            self.radar_front.enable(self.RADAR_PERIOD_MS)
            self.radar_sensor = RadarDetection(radar_device=self.radar_front,
                                               radar_relative_pos=(self.RADAR_FRONT_RELATIVE_POSITION[0],
                                                                   self.RADAR_FRONT_RELATIVE_POSITION[1]))
            self.perception_system.register_sensor(sensor_detector=self.radar_sensor,
                                                   sensor_period=self.RADAR_PERIOD_MS)
        self.sensor_display = self.getDisplay(self.sensor_display_device_name)
        self.sensor_visualizer = SensorVisualization(sensor_display=self.sensor_display,
                                                     ego_state=self.ego_state,
                                                     object_detector=self.perception_system,
                                                     lidar_device=self.lidar_main,
                                                     radar_device=self.radar_front)
        self.sensor_visualizer.set_camera_display(self.camera_info_display)
        if self.lidar_main is not None or self.radar_front is not None:
            self.high_level_controller.set_parameter('risky_obj_distance_threshold', 15.0)
        self.radio_comm_module = CommunicationModule(controller=self)
        # Start the car engine
        self.start_car()

    def run(self):
        """Runs the controller."""
        self.start_devices()
        print("INFO: It is normal to get device not found warnings for some sensors that are not part of the experiment.")
        print("Devices Started.")
        sys.stdout.flush()

        global data_dict

        while self.step() >= 0:
            sim_time = self.get_sim_time()
            cur_time_ms = int(round(1000 * sim_time))
            

            self.ego_state.update_states(cur_time_ms)
            # ************ Sensor fusion for detections ************
            # self.perception_system.update_detections(cur_time_ms)

            # Read sensor-like information and path updates from Simulation Supervisor
            self.radio_comm_module.receive_and_update(cur_time_ms)

            # self.path_planner.update_estimations(self.ego_state.get_position(),
            #                                      self.ego_state.get_speed_ms(),
            #                                      self.ego_state.get_yaw_angle(),
            #                                      self.perception_system.new_detections)

            (control_throttle, control_steering) = \
                self.high_level_controller.compute_control(self.perception_system.new_detections)
            
            # print("target_speed_m_s")
            # print(self.high_level_controller.target_speed_m_s)
            


            if cur_time_ms%1000==0:
                print(str(cur_time_ms)+" "+str(self.ego_state.get_speed_ms())+" "+str(self.ego_state.get_position()))
            

            
            ## Changed part
            global time_index
            if cur_time_ms<3010:
                x = 0.0
                self.set_target_speed_and_angle(speed=x,angle=control_steering)
                control_throttle = 0.0
            else:
                if(target_t[time_index] <= ((cur_time_ms/1000.0) -3) ):
                    time_index = time_index + 1
                    # x = controller_commons.speed_ms_to_kmh(target_v[time_index])
                # cur_v = target_v[time_index]
                # cur_a = target_a[time_index]
                self.set_throttle_and_steering_angle(target_throttle[time_index], control_steering)
                control_throttle = target_throttle[time_index]

            if cur_time_ms%100==0:
                global img_cnt
                img_name = "img_"+str(img_cnt)+".png"
                self.camera.saveImage("../../../images/"+img_name,1)
                img_cnt = img_cnt + 1
                data_dict[img_name] = [self.ego_state.get_speed_ms(),target_throttle[time_index],control_steering]

            ## Changed part ends here --------------------------------------------------------------


            # print(control_throttle)
            # self.set_throttle_and_steering_angle(control_throttle, control_steering)
            # self.set_throttle_and_steering_angle(1.0, control_steering)

            '''
            if self.detection_perf_monitor is not None:
                self.detection_perf_monitor.evaluate_detections()

            if self.visibility_monitor is not None:
                self.visibility_monitor.update_visibility_dict()
            '''
            # if SENSOR_TYPE == 'Perfect':
            #     control_steering = perfect_sens_control_steering
            #     control_throttle = perfect_sens_control_throttle
            #     self.very_risky_object_list = perf_sens_v_r_list
            #     self.risky_object_list = perf_sens_r_list
            #     self.proceed_w_caution_object_list = perf_sens_p_c_list

            # ------------------ Display Sensor Information ---------------------------
            # self.sensor_visualizer.update_sensor_display(cur_time_ms,
            #                                              control_throttle,
            #                                              control_steering,
            #                                              self.high_level_controller.control_mode,
            #                                              self.path_planner.trajectory_estimation.ego_future)

            # Transmit control information for logging.
            self.radio_comm_module.transmit_control_data(control_throttle, control_steering)
            self.radio_comm_module.transmit_detection_evaluation_data()
            self.radio_comm_module.transmit_visibility_evaluation_data()

        out_file = "../../../control_throttle.pkl"
        
        with open(out_file, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Clean up
        del self.classifier
        del self.obj_tracker
        print("Bye!")
        sys.stdout.flush()
