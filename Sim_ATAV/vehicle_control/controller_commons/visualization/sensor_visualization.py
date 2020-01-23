"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from Sim_ATAV.vehicle_control.controller_commons.visualization.sensor_info_display import SensorInfoDisplay
from Sim_ATAV.classifier.classifier_interface.classifier import Classifier
from Sim_ATAV.vehicle_control.controller_commons.perception.object_creation.sensor_object import SensorObject
from Sim_ATAV.vehicle_control.controller_commons import controller_commons


# TODO: This class is highly coupled to other classes. Try to reduce the coupling.
# (On the other hand this is only for debugging)


class SensorVisualization(object):
    def __init__(self, sensor_display, ego_state, object_detector, radar_device, lidar_device):
        self.sensor_info_display = SensorInfoDisplay(sensor_display)
        self.sensor_display_device = sensor_display
        self.ego_state = ego_state
        self.object_detector = object_detector
        self.radar_device = radar_device
        self.lidar_device = lidar_device
        self.camera_info_display = None

    def set_camera_display(self, camera_display):
        self.camera_info_display = camera_display

    def update_sensor_display(self, cur_time_ms, control_throttle, control_steering, control_mode, self_future_pos):
        sensor_detected_objects = self.object_detector.get_detections()
        if self.sensor_info_display is not None:
            clear_screen = True
            self.sensor_info_display.set_display_y_range(60.0)
            self.sensor_info_display.set_display_offset(
                [self.sensor_info_display.display_width / 2, self.sensor_info_display.display_height * 3 / 4])

            # Display Radar information
            if self.object_detector.object_detector.has_radar:
                self.sensor_info_display.display_radar_targets(self.object_detector.object_detector.radar_targets,
                                                               self.radar_device,
                                                               clear_screen=clear_screen)
                clear_screen = False
            # Display Camera Detected Objects
            if self.object_detector.object_detector.has_camera:
                for camera_detected_object in self.object_detector.object_detector.camera_objects:
                    if camera_detected_object.object_type == SensorObject.OBJECT_CAR:
                        obj_disp_color = Classifier.CAR_BOX_COLOR_HEX
                    elif camera_detected_object.object_type == SensorObject.OBJECT_PEDESTRIAN:
                        obj_disp_color = Classifier.PEDESTRIAN_BOX_COLOR_HEX
                    else:
                        obj_disp_color = Classifier.CYCLIST_BOX_COLOR_HEX
                    self.sensor_info_display.display_detection_object(camera_detected_object,
                                                                      obj_disp_color,
                                                                      clear_screen=clear_screen,
                                                                      radius=7)
                    clear_screen = False

            # # Display received ground truth information
            # for sensor_detected_object in perf_sensor_detected_objects:
            #     obj_disp_color = 0xFFFFFF
            #     self.sensor_info_display.display_detection_object(sensor_detected_object,
            #                                                       obj_disp_color,
            #                                                       clear_screen=clear_screen)
            #     clear_screen = False

            # Display LIDAR information
            if self.object_detector.object_detector.has_lidar:
                self.sensor_info_display.display_lidar_layer_point_cloud(
                    self.object_detector.object_detector.lidar_point_clouds,
                    self.lidar_device,
                    clear_screen=clear_screen,
                    pc_clusters=self.object_detector.object_detector.lidar_clusters,
                    display_all_points=True)
                clear_screen = False

            # Display GPS and Compass information
            self.sensor_info_display.display_speed_text(
                int(controller_commons.speed_ms_to_kmh(self.ego_state.get_speed_ms())),
                'km/h',
                clear_screen=clear_screen)
            clear_screen = False
            self.sensor_info_display.display_position_text(self.ego_state.get_position())
            self.sensor_info_display.display_bearing(self.ego_state.get_yaw_angle())

            for (obj_ind, detected_object) in enumerate(sensor_detected_objects):
                temp_pos = [-1 * detected_object.object_position[0], detected_object.object_position[1]]
                if detected_object.risk_level == SensorObject.HIGH_RISK:
                    self.sensor_info_display.display_text_at('x', temp_pos, text_color=0xFF0000)
                elif detected_object.risk_level == SensorObject.RISKY:
                    self.sensor_info_display.display_text_at('!', temp_pos, text_color=0xFFFF00)
                elif detected_object.risk_level == SensorObject.CAUTION:
                    self.sensor_info_display.display_text_at('+', temp_pos, text_color=0x00FF00)
                else:
                    self.sensor_info_display.display_text_at('^', temp_pos, text_color=0xFF00FF)

                if detected_object.object_type == SensorObject.OBJECT_CAR:
                    obj_disp_color = Classifier.CAR_BOX_COLOR_HEX
                elif detected_object.object_type == SensorObject.OBJECT_PEDESTRIAN:
                    obj_disp_color = Classifier.PEDESTRIAN_BOX_COLOR_HEX
                else:
                    obj_disp_color = Classifier.CYCLIST_BOX_COLOR_HEX
                self.sensor_info_display.display_detection_object(detected_object,
                                                                  obj_disp_color,
                                                                  clear_screen=clear_screen,
                                                                  fill_object=True,
                                                                  display_history=True,
                                                                  cur_sim_time=cur_time_ms)
                clear_screen = False
            self.sensor_info_display.display_future_trajectory(self_future_pos)
            self.sensor_info_display.display_legend()
            if self.sensor_display_device is not None:
                self.sensor_display_device.setColor(0xFFFFFF)
                self.sensor_display_device.drawText(control_mode,
                                                    int(0.75 * self.sensor_info_display.display_width),
                                                    int(self.sensor_info_display.display_height - 12))

                # self.sensor_display.drawText('{} kmh'.format(int(computed_2d_speed_m_s*3.6)),
                #                                             int(self.sensor_display.getWidth()/2)+10,
                #                                             int(self.sensor_display.getHeight()/2))

        # ------------------ Display Camera Information ---------------------------
        if self.camera_info_display is not None:
            camera_detected_objects = self.object_detector.object_detector.camera_objects
            if self.object_detector.object_detector.is_camera_read:
                self.camera_info_display.refresh_display_overlay()
                for (obj_ind, sensor_camera_detected_object) in enumerate(camera_detected_objects):
                    object_camera_data = sensor_camera_detected_object.sensor_aux_data_dict[SensorObject.SENSOR_CAMERA]
                    self.camera_info_display.show_object_detection_box(
                        object_camera_data.object_class,
                        object_camera_data.detection_box,
                        obj_ind=obj_ind,
                        is_tracker_detection=object_camera_data.added_by_tracker)
                for (obj_ind, detected_object) in enumerate(sensor_detected_objects):
                    if SensorObject.SENSOR_CAMERA in detected_object.sensor_aux_data_dict:
                        if detected_object.risk_level == SensorObject.HIGH_RISK:
                            object_camera_data = detected_object.sensor_aux_data_dict[SensorObject.SENSOR_CAMERA]
                            self.camera_info_display.mark_critical_object(object_camera_data.detection_box)
                        elif detected_object.risk_level == SensorObject.RISKY:
                            object_camera_data = detected_object.sensor_aux_data_dict[SensorObject.SENSOR_CAMERA]
                            self.camera_info_display.mark_critical_object(object_camera_data.detection_box,
                                                                          color=0xFFFF00)
                        elif detected_object.risk_level == SensorObject.CAUTION:
                            object_camera_data = detected_object.sensor_aux_data_dict[SensorObject.SENSOR_CAMERA]
                            self.camera_info_display.mark_critical_object(object_camera_data.detection_box,
                                                                          color=0x00FF00)
            self.camera_info_display.show_control_information(control_throttle, control_steering,
                                                              control_mode=control_mode)
