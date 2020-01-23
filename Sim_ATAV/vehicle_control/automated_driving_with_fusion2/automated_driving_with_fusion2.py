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

WORLD_TIME_STEP_MS = 10
HAS_DEBUG_DISPLAY = True
SENSOR_TYPE = 'Actual'  # 'Actual' #'Perfect'
DEBUG_MODE = False


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
        self.high_level_controller.set_parameter('target_speed_m_s', 50.0)
        self.radio_comm_module = None
        self.detection_perf_monitor = None
        self.visibility_monitor = None
        print('AutomatedDrivingControl Initialized: {}, {}'.format(car_model, self.target_speed_m_s))

    def start_devices(self):
        """Start the devices on the car and initialize objects like classifier."""
        # Start camera and the sensors:
        self.camera = self.getCamera(self.camera_device_name)
        if self.camera is not None:
            self.camera.enable(self.CLASSIFIER_PERIOD_MS)
            self.classifier = Classifier(is_show_image=False, is_gpu=self.has_gpu, processor_id=self.processor_id)
            self.classifier.start_classification_engine()
            # self.obj_tracker = ObjectTracker()
            self.ground_truth_generator = GroundTruthGenerator()
            self.ground_truth_generator.set_camera_parameters(self.camera.getWidth(),
                                                              self.camera.getHeight(),
                                                              self.camera.getFov())
            self.camera_sensor = CameraDetection(camera_device=self.camera,
                                                 classifier=self.classifier,
                                                 cam_relative_pos=(self.CAMERA_MAIN_RELATIVE_POSITION[0],
                                                                   self.CAMERA_MAIN_RELATIVE_POSITION[1]))
            self.perception_system.register_sensor(sensor_detector=self.camera_sensor,
                                                   sensor_period=self.CLASSIFIER_PERIOD_MS)
        self.display = self.getDisplay(self.display_device_name)
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

        while self.step() >= 0:
            sim_time = self.get_sim_time()
            cur_time_ms = int(round(1000 * sim_time))
            
            self.ego_state.update_states(cur_time_ms)
            # ************ Sensor fusion for detections ************
            self.perception_system.update_detections(cur_time_ms)

            # Read sensor-like information and path updates from Simulation Supervisor
            self.radio_comm_module.receive_and_update(cur_time_ms)

            self.path_planner.update_estimations(self.ego_state.get_position(),
                                                 self.ego_state.get_speed_ms(),
                                                 self.ego_state.get_yaw_angle(),
                                                 self.perception_system.new_detections)

            (control_throttle, control_steering) = \
                self.high_level_controller.compute_control(self.perception_system.new_detections)
            if cur_time_ms%1000==0:
                print(str(cur_time_ms)+" "+str(self.ego_state.get_speed_ms())+" "+str(self.ego_state.get_position()))
                # print(control_throttle)
            self.set_throttle_and_steering_angle(control_throttle, control_steering)
            # self.set_throttle_and_steering_angle(1.0, control_steering)

            if self.detection_perf_monitor is not None:
                self.detection_perf_monitor.evaluate_detections()

            if self.visibility_monitor is not None:
                self.visibility_monitor.update_visibility_dict()

            # if SENSOR_TYPE == 'Perfect':
            #     control_steering = perfect_sens_control_steering
            #     control_throttle = perfect_sens_control_throttle
            #     self.very_risky_object_list = perf_sens_v_r_list
            #     self.risky_object_list = perf_sens_r_list
            #     self.proceed_w_caution_object_list = perf_sens_p_c_list

            # ------------------ Display Sensor Information ---------------------------
            self.sensor_visualizer.update_sensor_display(cur_time_ms,
                                                         control_throttle,
                                                         control_steering,
                                                         self.high_level_controller.control_mode,
                                                         self.path_planner.trajectory_estimation.ego_future)

            # Transmit control information for logging.
            self.radio_comm_module.transmit_control_data(control_throttle, control_steering)
            self.radio_comm_module.transmit_detection_evaluation_data()
            self.radio_comm_module.transmit_visibility_evaluation_data()

        # Clean up
        del self.classifier
        del self.obj_tracker
        print("Bye!")
        sys.stdout.flush()
