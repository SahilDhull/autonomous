"""Defines PathFollowingTools class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import math
import copy
import dubins
import numpy as np
import shapely.geometry as geom


class PathFollowingTools(object):
    """PathFollowingTools provides tools for path following computations."""
    def __init__(self, target_points=None):
        if target_points is None:
            self.target_path = None
            self.future_starting_point = None
            self.starting_point = None
            self.detour_path = None
            self.path_details = None
            self.future_target_path = None
            self.future_path_details = None
        else:
            self.starting_point = target_points[0]
            self.target_path = geom.LineString(target_points)
            self.detour_path = self.target_path
            self.smoothen_the_path()
            self.populate_the_path_with_details()

    def add_point_to_path(self, target_pt, location=-1):
        """Add a new point to the path. Handles creation of the path if there is no path."""
        if self.starting_point is None:
            self.starting_point = target_pt
        elif self.target_path is None:
            self.target_path = geom.LineString([self.starting_point, target_pt])
        elif self.target_path == []:
            self.target_path = geom.LineString([self.starting_point, target_pt])
        else:
            points = list(self.target_path.coords)
            if location == -1:
                points.append((target_pt[0], target_pt[1]))
            else:
                points.insert(location, (target_pt[0], target_pt[1]))
            self.target_path = geom.LineString(points)

    def add_future_point_to_path(self, target_pt, location=-1):
        """Add a new point to the path. Handles creation of the path if there is no path."""
        if self.future_starting_point is None:
            self.future_starting_point = target_pt
        elif self.future_target_path is None:
            self.future_target_path = geom.LineString([self.starting_point, target_pt])
        elif self.future_target_path == []:
            self.future_target_path = geom.LineString([self.starting_point, target_pt])
        else:
            points = list(self.future_target_path.coords)
            if location == -1:
                points.append((target_pt[0], target_pt[1]))
            else:
                points.insert(location, (target_pt[0], target_pt[1]))
            self.future_target_path = geom.LineString(points)

    def get_distance_and_angle_error(self, vhc_pos, vhc_bearing, last_segment_ind=0, is_detouring=False):
        """Compute shortest distance to the path and orientation error wrt the closest point on the path."""
        if is_detouring:
            target_path = self.detour_path
        else:
            target_path = self.target_path
        if target_path is not None:
            (_closest_line_segment_ind, line_segment_as_list, nearest_pos_on_path, _distance_to_end_of_segment) = \
                self.get_current_segment(vhc_pos, last_segment_ind, target_path=target_path)
            vhc_future_pos = [vhc_pos[0] - math.sin(vhc_bearing), vhc_pos[1] + math.cos(vhc_bearing)]
            distance_err = target_path.distance(geom.Point(vhc_pos))
            d = (nearest_pos_on_path.x - vhc_pos[0]) * (vhc_future_pos[1] - vhc_pos[1]) - \
                (nearest_pos_on_path.y - vhc_pos[1]) * (vhc_future_pos[0] - vhc_pos[0])
            if d > 0:
                distance_err = -distance_err

            target_direction = math.atan2(-(line_segment_as_list[1][0] - line_segment_as_list[0][0]),
                                          line_segment_as_list[1][1] - line_segment_as_list[0][1])
            # - because +ve x is on the left in Webots.
            angle_err = target_direction - vhc_bearing
            while angle_err > math.pi:
                angle_err -= 2*math.pi
            while angle_err < -math.pi:
                angle_err += 2*math.pi
        else:
            angle_err = -vhc_bearing
            distance_err = -vhc_pos[1]
        return distance_err, angle_err

    def get_current_segment(self, vhc_pos, last_segment_ind=0, target_path=None):
        if target_path is None:
            target_path = self.target_path
        if target_path is not None:
            min_dist = math.inf
            nearest_pos_on_path = target_path.interpolate(target_path.project(geom.Point(vhc_pos)))
            min_line_segment = geom.LineString(target_path.coords[0:2])
            closest_line_segment_ind = None
            for pt_ind in range(len(target_path.coords) - 1):
                segment_ind = (pt_ind + last_segment_ind) % (len(target_path.coords) - 1)
                line_segment = geom.LineString(target_path.coords[segment_ind:segment_ind + 2])
                dist = nearest_pos_on_path.distance(line_segment)
                if dist < min_dist:
                    min_dist = dist
                    min_line_segment = line_segment
                    closest_line_segment_ind = segment_ind
                # close enough or started getting further away:
                if min_dist < 0.2 or (dist > min_dist + 2.0 and min_dist < 1.5):
                    min_line_segment = line_segment
                    closest_line_segment_ind = segment_ind
                    break
            if closest_line_segment_ind is None:
                closest_line_segment_ind = 0
            line_segment_as_list = list(min_line_segment.coords)
            distance_to_end_of_segment = math.sqrt(
                (line_segment_as_list[1][0] - vhc_pos[0]) ** 2 + (line_segment_as_list[1][1] - vhc_pos[1]) ** 2)
        else:
            closest_line_segment_ind = 0
            line_segment_as_list = []
            nearest_pos_on_path = [0.0, 0.0]
            distance_to_end_of_segment = 0.0
        return closest_line_segment_ind, line_segment_as_list, nearest_pos_on_path, distance_to_end_of_segment

    def get_next_turn(self, current_segment_ind, current_position, turn_angle_threshold=math.pi/60):
        """Compute turn angle, position and distance to the next turn on the path."""
        if self.target_path is not None and len(self.target_path.coords) > 0:
            turn_angle = 0.0
            turn_position = self.target_path.coords[-1]
            segment_ind = len(self.target_path.coords) - 1
            travel_distance = 0
            if len(self.target_path.coords) > 1:
                for pt_ind in range(current_segment_ind, len(self.target_path.coords) - 2):
                    line_segment = geom.LineString(self.target_path.coords[pt_ind:pt_ind + 2])
                    line_segment_as_list = list(line_segment.coords)
                    line_segment_as_vector = [line_segment_as_list[1][0] - line_segment_as_list[0][0],
                                              line_segment_as_list[1][1] - line_segment_as_list[0][1]]
                    cur_angle = math.atan2(line_segment_as_vector[0], line_segment_as_vector[1])

                    # Accumulate travel distance segment by segment
                    if pt_ind == current_segment_ind:
                        temp_travel_dist = math.sqrt((line_segment_as_list[1][0] - current_position[0])**2 +
                                                     (line_segment_as_list[1][1] - current_position[1])**2)
                    else:
                        temp_travel_dist = math.sqrt(line_segment_as_vector[0]**2 + line_segment_as_vector[1]**2)
                    travel_distance += temp_travel_dist

                    line_segment = geom.LineString(self.target_path.coords[pt_ind+1:pt_ind + 3])
                    line_segment_as_list = list(line_segment.coords)
                    line_segment_as_vector = [line_segment_as_list[1][0] - line_segment_as_list[0][0],
                                              line_segment_as_list[1][1] - line_segment_as_list[0][1]]
                    next_angle = math.atan2(line_segment_as_vector[0], line_segment_as_vector[1])
                    angle_diff = next_angle - cur_angle
                    if abs(angle_diff) > turn_angle_threshold:
                        if abs(turn_angle + angle_diff) > abs(turn_angle):
                            turn_angle = turn_angle + angle_diff
                            turn_position = line_segment_as_list[0]
                            segment_ind = pt_ind
                        else:
                            break
        else:
            turn_angle = 0.0
            turn_position = [0.0, 0.0]
            segment_ind = 0
            travel_distance = 0
        return turn_angle, turn_position, segment_ind, travel_distance

    def generate_dubins_path(self, start_pt, end_pt, turn_radius=10.0, step_size=1.0):
        """Generate points along a Dubins path connecting start point to end point.
        Format for input / output points: (x, y, angle)"""
        min_turn_radius = min(0.1, turn_radius)
        satisfied = False
        configurations = [start_pt, end_pt]
        while not satisfied:
            dubins_path = dubins.shortest_path(start_pt, end_pt, turn_radius)
            configurations, _ = dubins_path.sample_many(step_size)
            cex_found = False
            for configuration in configurations:
                if not (min(start_pt[0], end_pt[0]) - 0.1 <= configuration[0] <= max(start_pt[0], end_pt[0]) + 0.1 and
                        min(start_pt[1], end_pt[1]) - 0.1 <= configuration[1] <= max(start_pt[1], end_pt[1]) + 0.1):
                    cex_found = True
                    break
            satisfied = not cex_found
            if cex_found:
                # Decrease radius until finding a satisfying result.
                # We could do a binary search but that requires a termination condition.
                turn_radius = turn_radius*0.9
                if turn_radius < min_turn_radius:
                    break
        if not satisfied:
            configurations = [start_pt, end_pt]
        return configurations, satisfied

    def plot_dubins_path(self, configurations):
        """Plots given Dubins path."""
        import matplotlib.pyplot as plt
        qs = np.array(configurations)
        xs = qs[:, 0]
        ys = qs[:, 1]
        us = xs + np.cos(qs[:, 2])
        vs = ys + np.sin(qs[:, 2])
        plt.plot(xs, ys, 'b-')
        plt.plot(xs, ys, 'r.')
        for i in range(qs.shape[0]):
            plt.plot([xs[i], us[i]], [ys[i], vs[i]], 'r-')
        ax = plt.gca()
        # expand_plot(ax)
        ax.set_aspect('equal')
        plt.show()
    
    def plot_all_path(self):
        """Plots target path."""
        import matplotlib.pyplot as plt
        coords = np.array(self.target_path.coords)
        plt.plot(-coords[:, 0], coords[:, 1], 'r-')
        plt.axis('equal')
        plt.show()

    def convert_point_for_dubins_computation(self, point_coordinates, angle):
        """Converts the format and angle to the format accepted by the dubins compuations."""
        dub_angle = math.pi/2 - angle
        return point_coordinates[0], point_coordinates[1], dub_angle

    def convert_dubins_configurations_to_waypoints(self, configurations):
        """Converts Dubins configurations to list of x,y waypoints."""
        wpts = []
        for configuration in configurations:
            wpts.append([configuration[0], configuration[1]])
        return wpts

    def smoothen_the_path(self, turn_radius=10.0, step_size=1.0):
        """Converts linear segments to Dubins path where necessary."""
        new_segments = []
        if self.target_path is not None and len(self.target_path.coords) > 0:
            # First, connect turns with Dubins paths:
            num_original_points = len(self.target_path.coords)
            for pt_ind in range(num_original_points - 3):
                (init_line_segment_as_list, line_segment_as_vector, cur_angle, segment_length) = \
                    self.get_line_segment(pt_ind)
                (line_segment_as_list, line_segment_as_vector, next_angle, segment_length) = \
                    self.get_line_segment(pt_ind + 1)
                angle_diff = next_angle - cur_angle
                if abs(angle_diff) > math.pi / 60.0:
                    (line_segment_as_list, line_segment_as_vector, end_angle, segment_length) = \
                        self.get_line_segment(pt_ind + 2)
                    # If there is little angle difference or if the segment length is short,
                    # don't bother creating a Dubins path for this segments.
                    if (abs(next_angle - end_angle) > math.pi/60.0 and
                            np.linalg.norm(np.array(line_segment_as_list[0]) -
                                           np.array(init_line_segment_as_list[0])) > 3.0):
                        start_pt = self.convert_point_for_dubins_computation(
                            point_coordinates=init_line_segment_as_list[1], angle=cur_angle)
                        end_pt = self.convert_point_for_dubins_computation(
                            point_coordinates=line_segment_as_list[0], angle=end_angle)
                        (configurations, path_found) = self.generate_dubins_path(start_pt, end_pt,
                                                                                 turn_radius=turn_radius,
                                                                                 step_size=step_size)
                        if path_found:
                            new_points = self.convert_dubins_configurations_to_waypoints(configurations)
                            new_segments.append((pt_ind + 1, new_points))

            for new_segment in reversed(new_segments):
                insert_location = new_segment[0] + 1
                new_points = new_segment[1]
                if len(new_points) > 1:
                    for new_point in new_points[1:]:
                        self.add_point_to_path(new_point, location=insert_location)
                        insert_location += 1

    def smoothen_the_future_path(self, turn_radius=10.0, step_size=1.0):
        """Converts linear segments to Dubins path where necessary."""
        new_segments = []
        if self.future_target_path is not None and len(self.future_target_path.coords) > 0:
            # First, connect turns with Dubins paths:
            num_original_points = len(self.future_target_path.coords)
            for pt_ind in range(num_original_points - 3):
                (init_line_segment_as_list, line_segment_as_vector, cur_angle, segment_length) = \
                    self.get_future_line_segment(pt_ind)
                (line_segment_as_list, line_segment_as_vector, next_angle, segment_length) = \
                    self.get_future_line_segment(pt_ind + 1)
                angle_diff = next_angle - cur_angle
                if abs(angle_diff) > math.pi / 60.0:
                    (line_segment_as_list, line_segment_as_vector, end_angle, segment_length) = \
                        self.get_future_line_segment(pt_ind + 2)
                    # If there is little angle difference or if the segment length is short,
                    # don't bother creating a Dubins path for this segments.
                    if (abs(next_angle - end_angle) > math.pi/60.0 and
                            np.linalg.norm(np.array(line_segment_as_list[0]) -
                                           np.array(init_line_segment_as_list[0])) > 3.0):
                        start_pt = self.convert_point_for_dubins_computation(
                            point_coordinates=init_line_segment_as_list[1], angle=cur_angle)
                        end_pt = self.convert_point_for_dubins_computation(
                            point_coordinates=line_segment_as_list[0], angle=end_angle)
                        (configurations, path_found) = self.generate_dubins_path(start_pt, end_pt,
                                                                                 turn_radius=turn_radius,
                                                                                 step_size=step_size)
                        if path_found:
                            new_points = self.convert_dubins_configurations_to_waypoints(configurations)
                            new_segments.append((pt_ind + 1, new_points))

            for new_segment in reversed(new_segments):
                insert_location = new_segment[0] + 1
                new_points = new_segment[1]
                if len(new_points) > 1:
                    for new_point in new_points[1:]:
                        self.add_future_point_to_path(new_point, location=insert_location)
                        insert_location += 1

    def get_line_segment(self, segment_ind):
        """Returns the indexed segment as list, as vector and its angle and length."""
        line_segment = geom.LineString(self.target_path.coords[segment_ind:segment_ind + 2])
        line_segment_as_list = list(line_segment.coords)
        line_segment_as_vector = [line_segment_as_list[1][0] - line_segment_as_list[0][0],
                                  line_segment_as_list[1][1] - line_segment_as_list[0][1]]
        segment_angle = math.atan2(line_segment_as_vector[0], line_segment_as_vector[1])
        segment_length = math.sqrt(line_segment_as_vector[0]**2 + line_segment_as_vector[1]**2)
        return line_segment_as_list, line_segment_as_vector, segment_angle, segment_length

    def get_future_line_segment(self, segment_ind):
        """Returns the indexed segment as list, as vector and its angle and length."""
        line_segment = geom.LineString(self.future_target_path.coords[segment_ind:segment_ind + 2])
        line_segment_as_list = list(line_segment.coords)
        line_segment_as_vector = [line_segment_as_list[1][0] - line_segment_as_list[0][0],
                                  line_segment_as_list[1][1] - line_segment_as_list[0][1]]
        segment_angle = math.atan2(line_segment_as_vector[0], line_segment_as_vector[1])
        segment_length = math.sqrt(line_segment_as_vector[0]**2 + line_segment_as_vector[1]**2)
        return line_segment_as_list, line_segment_as_vector, segment_angle, segment_length

    def populate_the_path_with_details(self):
        """Compute turn angle, position and distance to the next turn for each segment on the path."""
        self.path_details = []
        if self.target_path is not None and len(self.target_path.coords) > 0:
            for current_segment_ind in range(len(self.target_path.coords) - 1):
                turn_angle = 0.0
                travel_distance = 0
                no_turn_distance = 0
                if current_segment_ind < len(self.target_path.coords) - 1:
                    turn_started = False
                    for pt_ind in range(current_segment_ind, len(self.target_path.coords) - 1):
                        (line_segment_as_list, line_segment_as_vector, cur_angle, segment_length) = \
                            self.get_line_segment(pt_ind)

                        # Accumulate travel distance segment by segment
                        if pt_ind != current_segment_ind and not turn_started:
                            travel_distance += segment_length

                        if pt_ind < len(self.target_path.coords) - 2:
                            (line_segment_as_list, line_segment_as_vector, next_angle, segment_length) = \
                                self.get_line_segment(pt_ind + 1)
                            angle_diff = next_angle - cur_angle
                            if abs(angle_diff) > math.pi/180.0:
                                if abs(turn_angle + angle_diff) > abs(turn_angle):
                                    turn_angle = turn_angle + angle_diff
                                    turn_started = True
                                    no_turn_distance = 0
                                else:
                                    break
                            else:
                                no_turn_distance += segment_length
                                if turn_started and no_turn_distance > 10.0:
                                    break
                        else:
                            turn_angle = math.pi
                else:  # last segment
                    travel_distance = 0.0
                    turn_angle = math.pi

                self.path_details.append((turn_angle, travel_distance))

    def populate_the_future_path_with_details(self):
        """Compute turn angle, position and distance to the next turn for each segment on the path."""
        self.future_path_details = []
        if self.future_target_path is not None and len(self.future_target_path.coords) > 0:
            for current_segment_ind in range(len(self.future_target_path.coords) - 1):
                turn_angle = 0.0
                travel_distance = 0
                no_turn_distance = 0
                if current_segment_ind < len(self.future_target_path.coords) - 1:
                    turn_started = False
                    for pt_ind in range(current_segment_ind, len(self.future_target_path.coords) - 1):
                        (line_segment_as_list, line_segment_as_vector, cur_angle, segment_length) = \
                            self.get_future_line_segment(pt_ind)

                        # Accumulate travel distance segment by segment
                        if pt_ind != current_segment_ind and not turn_started:
                            travel_distance += segment_length

                        if pt_ind < len(self.future_target_path.coords) - 2:
                            (line_segment_as_list, line_segment_as_vector, next_angle, segment_length) = \
                                self.get_future_line_segment(pt_ind + 1)
                            angle_diff = next_angle - cur_angle
                            if abs(angle_diff) > math.pi/180.0:
                                if abs(turn_angle + angle_diff) > abs(turn_angle):
                                    turn_angle = turn_angle + angle_diff
                                    turn_started = True
                                    no_turn_distance = 0
                                else:
                                    break
                            else:
                                no_turn_distance += segment_length
                                if turn_started and no_turn_distance > 10.0:
                                    break
                        else:
                            turn_angle = math.pi
                else:  # last segment
                    travel_distance = 0.0
                    turn_angle = math.pi

                self.future_path_details.append((turn_angle, travel_distance))

    def get_expected_travel_times_for_wpts(self, current_speed_m_s, current_position, current_segment_ind=0):
        """This makes a rough computation on time to reach to each waypoint.
        It assumes constant speed and perfect following of the path except the current segment."""
        wpt_arrival_times = []
        np_last_point = np.array(current_position)
        eta = 0.0

        if self.target_path is not None and len(self.target_path.coords) > current_segment_ind + 1:
            for (wpt_ind, wpt_coord) in enumerate(self.target_path.coords[current_segment_ind+1:]):
                np_next_point = np.array(wpt_coord)
                dist = np.linalg.norm(np_next_point - np_last_point)
                eta = eta + dist / current_speed_m_s
                wpt_arrival_times.append([wpt_ind, dist, eta])
                np_last_point = np_next_point[:]
        return wpt_arrival_times

    def get_expected_position_angle_at_time(self, target_time, current_position, current_speed_m_s,
                                            current_segment_ind=0, target_path=None, current_angle=None):
        """This makes a rough computation by using time to reach to each waypoint.
        It assumes constant speed and perfect following of the path except the current segment."""
        np_last_point = np.array(current_position)
        time_at_last_pt = 0.0
        expected_pos = np_last_point
        expected_angle = current_angle if current_angle is not None else 0.0
        eta = 0.0
        if target_path is None:
            target_path = self.target_path

        if target_path is not None and len(target_path.coords) > current_segment_ind + 1:
            for wpt_coord in target_path.coords[current_segment_ind+1:]:
                np_next_point = np.array(wpt_coord)
                dist = np.linalg.norm(np_next_point - np_last_point)
                eta += dist / max(current_speed_m_s, 1.0)
                if eta < target_time:
                    time_at_last_pt = eta
                    np_last_point = np_next_point[:]
                    expected_pos = np_last_point[:]
                else:
                    remaining_time = target_time - time_at_last_pt
                    leg_time = eta - time_at_last_pt
                    if leg_time < 0.00001:
                        expected_pos = np_last_point
                    else:
                        expected_pos = np_last_point + (np_next_point - np_last_point) * (remaining_time / leg_time)
                    break
            temp_vector = np_next_point - np_last_point
            expected_angle = math.atan2(-temp_vector[0], temp_vector[1])
        return expected_pos[:], expected_angle

    def get_detour_path(self, lat_shift, detour_length, current_position, current_orientation, current_segment_ind=0):
        rotation_matrix_cw = np.array([[math.cos(current_orientation), math.sin(current_orientation)],
                                       [-math.sin(current_orientation), math.cos(current_orientation)]])
        detour_path = []
        np_last_point = np.array(current_position)
        np_new_point = np_last_point + np.array([lat_shift, 0])
        np_new_point = np.array([[np_new_point[0]], [np_new_point[1]]])
        np_new_point = np.dot(rotation_matrix_cw, np_new_point)
        detour_path.append([np_new_point[0][0], np_new_point[1][0]])
        detour_ended = False
        if self.target_path is not None and len(self.target_path.coords) > current_segment_ind + 1:
            for wpt_coord in self.target_path.coords[current_segment_ind+1:]:
                np_next_point = np.array(wpt_coord)
                temp_vector = np_next_point - np_last_point
                temp_angle = math.atan2(-temp_vector[0], temp_vector[1])
                rotation_matrix_cw = np.array([[math.cos(temp_angle), math.sin(temp_angle)],
                                               [-math.sin(temp_angle), math.cos(temp_angle)]])
                dist = np.linalg.norm(temp_vector)
                if not detour_ended and dist < detour_length:
                    np_new_point = np_next_point + np.array([lat_shift, 0])
                    np_new_point = np.array([[np_new_point[0]], [np_new_point[1]]])
                    np_new_point = np.dot(rotation_matrix_cw, np_new_point)
                    detour_path.append([np_new_point[0][0], np_new_point[1][0]])
                    detour_length = detour_length - dist
                elif not detour_ended:
                    ratio = dist / detour_length
                    temp_point = np_last_point + ratio * (np_next_point - np_last_point)
                    np_new_point = temp_point - np.array([lat_shift, 0])
                    np_new_point = np.array([[np_new_point[0]], [np_new_point[1]]])
                    np_new_point = np.dot(rotation_matrix_cw, np_new_point)
                    detour_path.append([np_new_point[0][0], np_new_point[1][0]])
                    detour_length = 0
                    detour_ended = True
                else:
                    detour_path.append(wpt_coord[:])
        new_detour_path = geom.LineString(detour_path)
        print('---')
        for wpt_coord in new_detour_path.coords:
            print('detour: {}'.format(wpt_coord))
        return geom.LineString(detour_path)

    def set_detour_path(self, detour_path):
        self.detour_path = copy.deepcopy(detour_path)
