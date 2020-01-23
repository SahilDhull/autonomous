"""Defines SensorInfoDisplay class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import math
import numpy as np


class SensorInfoDisplay(object):
    """SensorInfoDisplay class handles common functions for visualizing sensor information."""

    def __init__(self, display_device):
        self.display = display_device
        self.display_y_range = 100.0
        if self.display is not None:
            self.display_width = self.display.getWidth()
            self.display_height = self.display.getHeight()
            self.set_display_offset([self.display_width / 2.0, self.display_height / 2.0])
        else:
            self.display_width = 0
            self.display_height = 0
            self.display_offset_x = 0
            self.display_offset_y = 0

    def set_display_offset(self, offset):
        """Set the offset (center) of the display.
        This typically corresponds to the ego vehicle position."""
        self.display_offset_x = int(offset[0])
        self.display_offset_y = int(offset[1])

    def set_display_y_range(self, display_y_range):
        """Sets the range in y from the center to the top of the display."""
        self.display_y_range = display_y_range

    def clear_display(self, background_color=0x000000):
        """Empty the Display."""
        if self.display is not None:
            # Remove objects:
            self.display.setColor(background_color)
            self.display.fillRectangle(0, 0, self.display_width, self.display_height)

    def put_ego_vehicle_icon(self, offset=None, icon_x=None, icon_y=None, icon_color=0xFFFF00):
        """Places an icon representing the ego vehicle in the below middle of the display."""
        if self.display is not None:
            if icon_x is not None and icon_y is not None:
                ego_vhc_icon_x = icon_x
                ego_vhc_icon_y = icon_y
            else:
                ego_vhc_icon_x = [0, 8, 0, -8]
                ego_vhc_icon_y = [0, 16, 12, 16]
            if offset is not None:
                self.set_display_offset(offset)
            self.display.setColor(icon_color)
            self.display.fillPolygon([x + self.display_offset_x for x in ego_vhc_icon_x],
                                     [-y + self.display_offset_y for y in ego_vhc_icon_y])
            self.display.setColor(0x000000)
            self.display.drawPixel(self.display_offset_x, self.display_offset_y)

    def generate_ego_vehicle_icon(self, display_scaling_factor=None, vehicle_front_length=3.6, vehicle_rear_length=0.7,
                                  vehicle_side_length=0.85):
        """Generate a rectangular icon for the Ego vehicle in correspondance with Display scaling and vhc dimensions."""
        if display_scaling_factor is None:
            display_max_range = self.display_y_range
            display_scaling_factor = self.display_offset_y / display_max_range
        ego_vhc_icon_x = [-vehicle_side_length * display_scaling_factor,
                          vehicle_side_length * display_scaling_factor,
                          vehicle_side_length * display_scaling_factor,
                          -vehicle_side_length * display_scaling_factor]
        ego_vhc_icon_y = [vehicle_front_length * display_scaling_factor,
                          vehicle_front_length * display_scaling_factor,
                          -vehicle_rear_length * display_scaling_factor,
                          -vehicle_rear_length * display_scaling_factor]
        return [ego_vhc_icon_x, ego_vhc_icon_y]

    def display_lidar_range_image(self, lidar_range_image, lidar_device):
        """Display LIDAR Range Image."""
        if lidar_device is not None and self.display is not None:
            self.clear_display()
            self.put_ego_vehicle_icon()
            lidar_hor_res = lidar_device.getHorizontalResolution()
            lidar_fov = lidar_device.getFov()
            lidar_angle_step = lidar_device.getFov() / lidar_hor_res
            lidar_max_range = lidar_device.getMaxRange()
            cur_angle = -lidar_fov/2
            self.display.setColor(0xFFFFFF)
            for layer in [60]:  # range(64):
                for i in range(lidar_hor_res*layer, lidar_hor_res*(layer+1)):
                    pt_x = lidar_range_image[i] * math.sin(cur_angle)
                    pt_y = lidar_range_image[i] * math.cos(cur_angle)
                    int_x = self.display_offset_x + int(pt_x)
                    int_y = self.display_offset_y + int(-lidar_max_range/2 - pt_y)
                    if 0 < int_x < self.display_width and 0 < int_y < self.display_height:
                        self.display.drawPixel(int_x, int_y)
                    cur_angle += lidar_angle_step

    def display_lidar_layer_point_cloud(self,
                                        lidar_layer_point_cloud,
                                        lidar_device,
                                        lidar_color=0xFFFFFF,
                                        background_color=0x000000,
                                        clear_screen=False,
                                        display_y_range=None,
                                        display_all_points=False,
                                        pc_clusters=None):
        """Display LIDAR Point Cloud."""
        if pc_clusters is None:
            pc_clusters = []
        if lidar_device is not None and self.display is not None:
            lidar_max_range = lidar_device.getMaxRange()
            if display_y_range is None:
                display_max_range = self.display_y_range
            else:
                display_max_range = display_y_range
            display_scaling_factor = self.display_offset_y / display_max_range
            vhc_center_to_lidar_px = 1.3 * display_scaling_factor
            if clear_screen:
                self.clear_display(background_color=background_color)
                vhc_icon = self.generate_ego_vehicle_icon(display_scaling_factor)
                self.put_ego_vehicle_icon(icon_x=vhc_icon[0], icon_y=vhc_icon[1])

            # Put points:
            if display_all_points:
                self.display.setColor(lidar_color)
                for lidar_point in lidar_layer_point_cloud:
                    if lidar_point.x <= lidar_max_range and lidar_point.z <= lidar_max_range:
                        disp_x = int(-lidar_point.x * display_scaling_factor + self.display_offset_x)
                        disp_y = int(-lidar_point.z * display_scaling_factor + self.display_offset_y)
                        if 0 <= disp_x <= self.display_width and 0 <= disp_y <= self.display_height:
                            self.display.drawPixel(disp_x, disp_y)

            # Draw rectangles for clusters:
            n_clusters = len(pc_clusters)
            if n_clusters > 0:
                for cluster in pc_clusters:
                    self.display.setColor(0x0000FF)
                    rect_x = int(-cluster.max_x * display_scaling_factor + self.display_offset_x)
                    rect_y = int(-cluster.max_y * display_scaling_factor + self.display_offset_y
                                 - vhc_center_to_lidar_px)
                    if 0 < rect_x < self.display_width and 0 < rect_y < self.display_height:
                        rect_w = max(int((cluster.max_x - cluster.min_x) * display_scaling_factor), 2)
                        rect_w = min(rect_w, int(self.display_width - 1 - rect_x))
                        rect_h = max(int((cluster.max_y - cluster.min_y) * display_scaling_factor), 2)
                        rect_h = min(rect_h, int(self.display_height - 1 - (rect_y + rect_h)))
                        if rect_w > 0:
                            self.display.drawRectangle(rect_x, rect_y, rect_w, rect_h)

    def display_radar_targets(self,
                              radar_targets,
                              radar_device,
                              background_color=0x000000,
                              clear_screen=False,
                              display_y_range=None):
        """Display the radar targets as boxes that are sized and colored based on the radar power."""
        if radar_device is not None and self.display is not None:
            radar_max_range = radar_device.getMaxRange()
            if display_y_range is None:
                display_max_range = self.display_y_range
            else:
                display_max_range = display_y_range
            display_scaling_factor = self.display_offset_y / display_max_range
            vhc_icon = self.generate_ego_vehicle_icon(display_scaling_factor)
            if clear_screen:
                self.clear_display(background_color=background_color)
                self.put_ego_vehicle_icon(icon_x=vhc_icon[0], icon_y=vhc_icon[1])
            # Draw Radar range lines
            vhc_center_to_radar_px = vhc_icon[1][0]
            radar_hor_fov = radar_device.getHorizontalFov()
            self.display.setColor(0xFF00FF)
            self.display.drawLine(self.display_offset_x, int(self.display_offset_y - vhc_center_to_radar_px),
                                  int(self.display_offset_x + int((self.display_offset_y - vhc_center_to_radar_px)
                                                                  * math.tan(-radar_hor_fov/2))), 0)
            self.display.drawLine(self.display_offset_x, int(self.display_offset_y - vhc_center_to_radar_px),
                                  int(self.display_offset_x + int((self.display_offset_y - vhc_center_to_radar_px)
                                                                  * math.tan(radar_hor_fov/2))), 0)

            # Draw radar targets
            radar_min_range = radar_device.getMinRange()
            for radar_target in radar_targets:
                # Right at min_range it detects the car itself:
                if radar_min_range < radar_target.distance < radar_max_range:
                    # Following is adapted from radar example code provided with Webots
                    target_x = radar_target.distance * math.sin(radar_target.azimuth)
                    target_z = -radar_target.distance

                    color_factor = (radar_target.received_power + 60.0) / 50.0
                    color_factor = max(min(color_factor, 1), 0)
                    color_factor = pow(color_factor, 0.33)  # linearize the value
                    color_str = '%02x%02x%02x' % (int(color_factor*255), 0, int((1-color_factor)*255))
                    target_color = int(color_str, 16)
                    # Get original received power in order to guess which type of object it is
                    power = pow(10, radar_target.received_power / 10.0) * 0.001 * pow(radar_target.distance, 4.0)
                    if power < 0.005:  # Motorcycle
                        target_dims_2d = [1.0*(power/0.005), 3.0*(power/0.005)]
                    elif power > 0.015:  # Large vehicle (bus)
                        target_dims_2d = [3.5*(power/0.015), 10.0*(power/0.015)]
                    else:  # Car
                        target_dims_2d = [1.8*(power/0.010), 4.5*(power/0.010)]
                    target_shape_x = [(-target_dims_2d[0] / 2 + target_x) * display_scaling_factor,
                                      (target_dims_2d[0] / 2 + target_x) * display_scaling_factor,
                                      (target_dims_2d[0] / 2 + target_x) * display_scaling_factor,
                                      (-target_dims_2d[0] / 2 + target_x) * display_scaling_factor]
                    target_shape_y = [(target_dims_2d[1] / 2 + target_z) * display_scaling_factor
                                      - vhc_center_to_radar_px,
                                      (target_dims_2d[1] / 2 + target_z) * display_scaling_factor
                                      - vhc_center_to_radar_px,
                                      (-target_dims_2d[1] / 2 + target_z) * display_scaling_factor
                                      - vhc_center_to_radar_px,
                                      (-target_dims_2d[1] / 2 + target_z) * display_scaling_factor
                                      - vhc_center_to_radar_px]
                    self.display.setColor(target_color)
                    self.display.fillPolygon([x + self.display_offset_x for x in target_shape_x],
                                             [y + self.display_offset_y for y in target_shape_y])

    def display_detection_object(self,
                                 detected_object,
                                 object_color=0x00FF00,
                                 background_color=0x000000,
                                 clear_screen=False,
                                 display_y_range=None,
                                 fill_object=False,
                                 radius=5,
                                 display_history=False,
                                 cur_sim_time=None,
                                 history_time_ms=1000.0):
        """Display detected object as a circle on the screen."""
        if self.display is not None:
            object_position = [-1*detected_object.object_position[0], detected_object.object_position[1]]
            if object_position:
                if detected_object.is_old_object:
                    object_color = 0x808080
                if display_y_range is None:
                    display_max_range = self.display_y_range
                else:
                    display_max_range = display_y_range
                display_scaling_factor = self.display_offset_y / display_max_range
                object_x = object_position[0] * display_scaling_factor
                object_y = object_position[1] * display_scaling_factor
                if clear_screen:
                    self.clear_display(background_color=background_color)
                    vhc_icon = self.generate_ego_vehicle_icon(display_scaling_factor)
                    self.put_ego_vehicle_icon(icon_x=vhc_icon[0], icon_y=vhc_icon[1])
                self.display.setColor(object_color)
                if fill_object:
                    self.display.fillOval(int(object_x + self.display_offset_x),
                                          int(-object_y + self.display_offset_y), 3, 3)
                else:
                    self.display.drawOval(int(object_x + self.display_offset_x),
                                          int(-object_y + self.display_offset_y), int(radius), int(radius))
                self.display.setColor(0xFFFFFF)
                if np.linalg.norm(detected_object.object_direction) > 0.01:
                    arrow = np.array(detected_object.object_direction)*detected_object.object_speed_m_s*2.0
                    arrow_tip_x = min(self.display_width, max(0, object_x - arrow[0] + self.display_offset_x))
                    arrow_tip_y = min(self.display_height, max(0, -object_y - arrow[1] + self.display_offset_y))
                    self.display.drawLine(int(object_x + self.display_offset_x),
                                          int(-object_y + self.display_offset_y),
                                          int(arrow_tip_x),
                                          int(arrow_tip_y))
            if display_history:
                # display object history:
                opacity = 1.0
                self.display.setColor(int(object_color*0.5))
                for history_obj in reversed(detected_object.history):
                    object_position = [-1*history_obj[0][0], history_obj[0][1]]
                    if object_position and (cur_sim_time is None or history_obj[3] > cur_sim_time-history_time_ms):
                        if display_y_range is None:
                            display_max_range = self.display_y_range
                        else:
                            display_max_range = display_y_range
                        display_scaling_factor = self.display_offset_y / display_max_range
                        object_x = object_position[0] * display_scaling_factor
                        object_y = object_position[1] * display_scaling_factor
                        opacity = opacity*0.9
                        self.display.setOpacity(opacity)
                        self.display.drawOval(int(object_x + self.display_offset_x),
                                              int(-object_y + self.display_offset_y), int(radius-1), int(radius-1))
                self.display.setOpacity(1.0)

                # display object future:
                opacity = 1.0
                self.display.setColor(object_color)
                for future_state in detected_object.future:
                    object_position = [-1*future_state[0], future_state[1]]
                    if object_position:
                        if display_y_range is None:
                            display_max_range = self.display_y_range
                        else:
                            display_max_range = display_y_range
                        display_scaling_factor = self.display_offset_y / display_max_range
                        object_x = object_position[0] * display_scaling_factor
                        object_y = object_position[1] * display_scaling_factor
                        opacity = opacity*0.9
                        self.display.setOpacity(opacity)
                        self.display.drawOval(int(object_x + self.display_offset_x),
                                              int(-object_y + self.display_offset_y), int(radius-1), int(radius-1))
                self.display.setOpacity(1.0)

    def display_future_trajectory(self, future_pos_list):
        """Display estimated future positions."""
        if self.display is not None:
            opacity = 1.0
            self.display.setColor(0xFFFF00)
            for future_pos in future_pos_list:
                display_max_range = self.display_y_range
                display_scaling_factor = self.display_offset_y / display_max_range
                object_x = -1.0 * future_pos[0] * display_scaling_factor
                object_y = future_pos[1] * display_scaling_factor
                opacity = opacity*0.9
                self.display.setOpacity(opacity)
                self.display.drawOval(int(object_x + self.display_offset_x),
                                      int(-object_y + self.display_offset_y), int(3), int(3))
            self.display.setOpacity(1.0)

    def display_speed_text(self,
                           speed_value,
                           speed_unit,
                           speed_color=0xFFFFFF,
                           background_color=0x000000,
                           clear_screen=False,
                           display_y_range=None):
        """Display speed as text on the upper left corner."""
        if self.display is not None:
            if clear_screen:
                if display_y_range is None:
                    display_max_range = self.display_y_range
                else:
                    display_max_range = display_y_range
                display_scaling_factor = self.display_offset_y / display_max_range
                self.clear_display(background_color=background_color)
                vhc_icon = self.generate_ego_vehicle_icon(display_scaling_factor)
                self.put_ego_vehicle_icon(icon_x=vhc_icon[0], icon_y=vhc_icon[1])
            self.display.setColor(speed_color)
            self.display.setFont('Arial', 15, False)
            self.display.drawText('{} {}'.format(speed_value, speed_unit), 5, 5)

    def display_position_text(self,
                              gps_position,
                              gps_position_color=0xFFFFFF,
                              background_color=0x000000,
                              clear_screen=False,
                              display_y_range=None):
        """Display gps_position as text on the lower left corner."""
        if self.display is not None:
            if clear_screen:
                if display_y_range is None:
                    display_max_range = self.display_y_range
                else:
                    display_max_range = display_y_range
                display_scaling_factor = self.display_offset_y / display_max_range
                self.clear_display(background_color=background_color)
                vhc_icon = self.generate_ego_vehicle_icon(display_scaling_factor)
                self.put_ego_vehicle_icon(icon_x=vhc_icon[0], icon_y=vhc_icon[1])
            self.display.setColor(gps_position_color)
            self.display.setFont('Arial', 8, False)
            if len(gps_position) > 2:
                self.display.drawText('{}, {}'.format(gps_position[0], gps_position[2]), 5, self.display_height - 25)
            else:
                self.display.drawText('{}, {}'.format(gps_position[0], gps_position[1]), 5, self.display_height - 25)

    def display_bearing(self,
                        bearing_value_rad,
                        bearing_color=0xFF0000,
                        bearing_background_color=0xFFFFFF,
                        background_color=0x000000,
                        clear_screen=False,
                        display_y_range=None):
        """Display bearing wrt North as arrow on the lower right corner."""
        if self.display is not None:
            if clear_screen:
                if display_y_range is None:
                    display_max_range = self.display_y_range
                else:
                    display_max_range = display_y_range
                display_scaling_factor = self.display_offset_y / display_max_range
                self.clear_display(background_color=background_color)
                vhc_icon = self.generate_ego_vehicle_icon(display_scaling_factor)
                self.put_ego_vehicle_icon(icon_x=vhc_icon[0], icon_y=vhc_icon[1])
            self.display.setColor(bearing_background_color)
            radial_center_x = int(self.display_width - 15)
            radial_center_y = int(self.display_height - 15)
            arrow_length = 12
            self.display.fillOval(radial_center_x, radial_center_y, 13, 13)
            self.display.setColor(bearing_color)
            self.display.drawLine(radial_center_x,
                                  radial_center_y,
                                  int(radial_center_x + arrow_length*math.sin(bearing_value_rad)),
                                  int(radial_center_y - arrow_length*math.cos(bearing_value_rad)))

    def display_text_at(self,
                        text,
                        position,
                        text_color=0xFF0000,
                        clear_screen=False,
                        display_y_range=None):
        """Display given text at given position. (Position is in world coordinates not display coordinates)"""
        if self.display is not None:
            if display_y_range is None:
                display_max_range = self.display_y_range
            else:
                display_max_range = display_y_range
            display_scaling_factor = self.display_offset_y / display_max_range

            vhc_icon = self.generate_ego_vehicle_icon(display_scaling_factor)
            if clear_screen:
                self.clear_display()
                self.put_ego_vehicle_icon(icon_x=vhc_icon[0], icon_y=vhc_icon[1])
            self.display.setColor(text_color)
            self.display.setFont('Arial', 10, False)
            self.display.drawText(text,
                                  int(position[0] * display_scaling_factor + self.display_offset_x),
                                  int(-position[1] * display_scaling_factor + self.display_offset_y))

    def display_legend(self):
        """Display legend for detections"""
        legend_x = int(self.display_width/2)
        legend_y = int(self.display_height-50)
        if self.display is not None:
            self.display.setFont('Arial', 8, False)
            self.display.setColor(0xFF0000)
            self.display.fillOval(legend_x, legend_y + 4, int(3), int(3))
            self.display.drawText('Decision', legend_x + 8, legend_y)
            self.display.setColor(0xFF0000)
            self.display.drawOval(legend_x, legend_y+14, int(5), int(5))
            self.display.drawText('Camera', legend_x + 8, legend_y+10)
            self.display.setColor(0xFFFFFF)
            self.display.drawOval(legend_x, legend_y+24, int(4), int(4))
            self.display.drawText('Grd. truth', legend_x + 8, legend_y+20)
            self.display.setColor(0x0000FF)
            self.display.drawRectangle(legend_x, legend_y+30, int(6), int(8))
            self.display.drawText('Lidar', legend_x + 8, legend_y+30)
            self.display.setColor(0xFF00FF)
            self.display.fillRectangle(legend_x, legend_y+40, int(6), int(8))
            self.display.drawText('Radar', legend_x + 8, legend_y+40)
            self.display.setColor(0xFFFFFF)
            self.display.drawRectangle(legend_x-8, legend_y-2, int(70), int(51))
