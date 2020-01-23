"""Defines SimObjectGenerator class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import math
from Sim_ATAV.simulation_control.webots_sensor import WebotsSensor
from Sim_ATAV.simulation_control.webots_road_disturbance import WebotsRoadDisturbance


class SimObjectGenerator(object):
    """SimObjectGenerator class translate between simulation objects
    and the strings declaring the objects in Webots."""
    VHC_DUMMY = 0
    VHC_VUT = 1

    def __init__(self):
        pass

    def generate_road_network_string(self, road_list, road_network_id):
        """Generates the Webots string for a list of road segments.
        These road segments go under a road segment."""
        road_network_string = "#VRML_OBJ R2018a utf8\n"
        road_network_string += "DEF ROAD_NETWORK_" + str(road_network_id) + " Solid {\n"
        road_network_string += "  children ["
        for i in range(0, len(road_list)):
            road_network_string += "    DEF " + \
                road_list[i].def_name + " " + road_list[i].road_type + " {"
            road_network_string += "      translation " + \
                str(road_list[i].position[0]) + " " + str(road_list[i].position[1]) + " " + \
                str(road_list[i].position[2]) + " " + "\n"
            road_network_string += "      rotation " + \
                str(road_list[i].rotation[0]) + " " + str(road_list[i].rotation[1]) + " " + \
                str(road_list[i].rotation[2]) + " " + str(road_list[i].rotation[3]) + " " + "\n"
            road_network_string += "      width " + str(road_list[i].width) + "\n"
            road_network_string += "      numberOfLanes " + str(road_list[i].number_of_lanes) + "\n"
            road_network_string += "      length " + str(road_list[i].length) + "\n"
            road_network_string += "      rightBorderBoundingObject " + \
                str(road_list[i].right_border_bounding_object).upper() + "\n"
            road_network_string += "      leftBorderBoundingObject " + \
                str(road_list[i].left_border_bounding_object).upper() + "\n"
            road_network_string += "    }\n"
        road_network_string += "  ]\n"
        road_network_string += "}\n"
        return road_network_string

    def generate_road_disturbance_string(self, road_disturbance):
        """Generates the Webots string for a road disturbance."""
        obj_string = "#VRML_OBJ R2018a utf8\n"
        obj_string += "DEF ROAD_DISTURBANCE_" + str(road_disturbance.disturbance_id) + " Solid {\n"
        rect_side = road_disturbance.height/math.sin(math.pi/8.0)  # Rectangle side size to give the desired height
        corner_elevation = math.sin(math.pi/8.0)*rect_side*math.sin(math.pi/4.0)
        obj_string += "  translation {} {} {}\n".format(road_disturbance.position[0],
                                                        road_disturbance.position[1] +
                                                            road_disturbance.surface_height -
                                                            corner_elevation,
                                                        road_disturbance.position[2])
        obj_string += "  rotation {} {} {} {}\n".format(road_disturbance.rotation[0],
                                                        road_disturbance.rotation[1],
                                                        road_disturbance.rotation[2],
                                                        road_disturbance.rotation[3])
        obj_string += "  children [\n"
        num_of_disturbances = int(road_disturbance.length / road_disturbance.inter_object_spacing)
        
        current_position = [0.0, 0.0, 0.0]
        current_side = 'right'
        for dist_ind in range(0, num_of_disturbances):
            current_position[2] = current_position[2] + road_disturbance.inter_object_spacing
            if (road_disturbance.disturbance_type == WebotsRoadDisturbance.TRIANGLE_LEFT_HALF or 
                 (road_disturbance.disturbance_type == WebotsRoadDisturbance.TRIANGLE_DOUBLE_SIDED 
                    and current_side == 'left')):
                current_position[0] = road_disturbance.width/4.0
                rect_length = road_disturbance.width/2.0
                current_side = 'right'
            elif (road_disturbance.disturbance_type == WebotsRoadDisturbance.TRIANGLE_RIGHT_HALF or 
                 (road_disturbance.disturbance_type == WebotsRoadDisturbance.TRIANGLE_DOUBLE_SIDED 
                    and current_side == 'right')):
                current_position[0] = -road_disturbance.width/4.0
                rect_length = road_disturbance.width/2.0
                current_side = 'left'
            else:
                current_position[0] = 0.0
                rect_length = road_disturbance.width
            obj_string += "    DEF ROAD_DIST_{}".format(dist_ind) + " Transform {\n"
            obj_string += "      translation {} {} {}\n".format(current_position[0],
                                                                current_position[1],
                                                                current_position[2])
            obj_string += "      rotation 1 0 0 {}\n".format(3.0*math.pi/8.0)
            obj_string += "      children [\n"
            if dist_ind == 0:  # Define shape once, repeat for others
                obj_string += "        DEF DISTURBANCE_SHAPE Shape {\n"
                obj_string += "          appearance Appearance {\n"
                obj_string += "            material Material {\n"
                obj_string += "              diffuseColor 0.2 0.2 0.2\n"
                obj_string += "            }\n"
                obj_string += "          }\n"
                obj_string += "          geometry Box {\n"
                obj_string += "            size {} {} {}\n".format(rect_length, rect_side, rect_side)
                obj_string += "          }\n"
                obj_string += "          castShadows FALSE\n"
                obj_string += "        }\n"
            else:
                obj_string += "        USE DISTURBANCE_SHAPE\n"
            obj_string += "      ]\n"
            obj_string += "    }\n"
        obj_string += "  ]\n"
        obj_string += "  boundingObject Group {\n"
        obj_string += "    children [\n"
        for dist_ind in range(0, num_of_disturbances):
            obj_string += "      USE ROAD_DIST_{}\n".format(dist_ind)
        obj_string += "    ]\n"
        obj_string += "  }\n"
        obj_string += "}\n"
        return obj_string

    def generate_fog_string(self, fog):
        """Generates the Webots string for fog."""
        obj_string = "#VRML_OBJ R2018a utf8\n"
        obj_string += "Fog {\n"
        obj_string += "  color {:.2f} {:.2f} {:.2f}".format(fog.color[0], fog.color[1], fog.color[2])
        obj_string += "  visibilityRange {:.1f}".format(fog.visibility_range)
        obj_string += "}\n"
        return obj_string

    def generate_vehicle_string(self, vhc_object):
        """Generates the Webots string for a vehicle."""
        vehicle_string = "#VRML_OBJ R2018a utf8\n"
        vehicle_string += "DEF " + vhc_object.def_name + " " + vhc_object.vehicle_model + " {\n"
        vehicle_string += "  translation " + \
            str(vhc_object.current_position[0]) + " " + str(vhc_object.current_position[1]) + \
            " " + str(vhc_object.current_position[2]) + " " + "\n"
        vehicle_string += "  rotation " + str(vhc_object.rotation[0]) + " " + \
            str(vhc_object.rotation[1]) + " " + str(vhc_object.rotation[2]) + \
            " " + str(vhc_object.rotation[3]) + " " + "\n"
        vehicle_string += "  color " + str(vhc_object.color[0]) + " " + \
            str(vhc_object.color[1]) + " " + str(vhc_object.color[2]) + "\n"
        vehicle_string += '  name \"' + vhc_object.def_name + '\"\n'
        for (param_name, param_val) in vhc_object.vehicle_parameters:
            vehicle_string += '  ' + param_name + ' ' + param_val + '\n'
        if not ('Simple' in vhc_object.vehicle_model):
            if vhc_object.is_controller_name_absolute:
                vehicle_string += "  controller \"" + vhc_object.controller + "\"\n"
                if vhc_object.controller_arguments:
                    vehicle_string += "  controllerArgs \""
                    for (arg_ind, argument_string) in enumerate(vhc_object.controller_arguments):
                        if arg_ind > 0:
                            vehicle_string += ' '
                        vehicle_string += argument_string
                    vehicle_string += "\"\n"
            else:
                vehicle_string += "  controller \"vehicle_controller\"\n"
                vehicle_string += "  controllerArgs \"" + \
                    vhc_object.controller + " " + vhc_object.vehicle_model
                for argument_string in vhc_object.controller_arguments:
                    vehicle_string += ' '
                    vehicle_string += argument_string
                vehicle_string += "\"\n"

            # Add text for Front Sensors if there is any
            vehicle_string += self.generate_sensor_string(vhc_object.sensor_array, WebotsSensor.FRONT)
            # Add text for Center Sensors if there is any
            vehicle_string += self.generate_sensor_string(vhc_object.sensor_array, WebotsSensor.CENTER)
            # Add text for Left Sensors if there is any
            vehicle_string += self.generate_sensor_string(vhc_object.sensor_array, WebotsSensor.LEFT)
            vehicle_string += \
                self.generate_sensor_string(vhc_object.sensor_array, WebotsSensor.LEFT_FRONT)
            vehicle_string += \
                self.generate_sensor_string(vhc_object.sensor_array, WebotsSensor.LEFT_REAR)
            # Add text for Right Sensors if there is any
            vehicle_string += self.generate_sensor_string(vhc_object.sensor_array, WebotsSensor.RIGHT)
            vehicle_string += \
                self.generate_sensor_string(vhc_object.sensor_array, WebotsSensor.RIGHT_FRONT)
            vehicle_string += \
                self.generate_sensor_string(vhc_object.sensor_array, WebotsSensor.RIGHT_REAR)
            # Add text for Rear Sensors if there is any
            vehicle_string += self.generate_sensor_string(vhc_object.sensor_array, WebotsSensor.REAR)
            # Add text for Top Sensors if there is any
            vehicle_string += self.generate_sensor_string(vhc_object.sensor_array, WebotsSensor.TOP)
        vehicle_string += "}\n"
        return vehicle_string

    def generate_pedestrian_string(self, sim_object):
        """Generates the Webots string for a pedestrian."""
        obj_string = '#VRML_OBJ R2018a utf8\n'
        obj_string += 'DEF ' + sim_object.def_name + ' Pedestrian {\n'
        obj_string += '  translation ' + \
            str(sim_object.current_position[0]) + ' ' + str(sim_object.current_position[1]) + \
            ' ' + str(sim_object.current_position[2]) + '\n'
        obj_string += '  rotation ' + str(sim_object.rotation[0]) + ' ' + \
            str(sim_object.rotation[1]) + ' ' + str(sim_object.rotation[2]) + \
            ' ' + str(sim_object.rotation[3]) + "\n"
        obj_string += '  name \"' + sim_object.def_name + '\"\n'
        #obj_string += '  abstractionLevel 1\n'
        obj_string += '  controller \"' + sim_object.controller + '\"\n'
        if sim_object.trajectory or sim_object.target_speed != 0.0:
            obj_string += '  controllerArgs \"'
            if sim_object.trajectory:
                obj_string += '--trajectory \\\"'
                for (pt_ind, traj_point) in enumerate(sim_object.trajectory):
                    obj_string += str(traj_point)
                    if pt_ind % 2 == 0:
                        obj_string += ' '
                    else:
                        if pt_ind < len(sim_object.trajectory) - 1:
                            obj_string += ', '
                obj_string += '\\\"'
            if sim_object.target_speed != 0.0:
                obj_string += ' --speed ' + str(sim_object.target_speed)
            obj_string += '\"\n'
        obj_string += '  shirtColor ' + \
            str(sim_object.shirt_color[0]) + ' ' + str(sim_object.shirt_color[1]) + \
            ' ' + str(sim_object.shirt_color[2]) + '\n'
        obj_string += '  pantsColor ' + \
            str(sim_object.pants_color[0]) + ' ' + str(sim_object.pants_color[1]) + \
            ' ' + str(sim_object.pants_color[2]) + '\n'
        obj_string += '  shoesColor ' + \
            str(sim_object.shoes_color[0]) + ' ' + str(sim_object.shoes_color[1]) + \
            ' ' + str(sim_object.shoes_color[2]) + '\n'
        obj_string += "}"
        return obj_string

    def generate_object_string(self, sim_object):
        """Generates the Webots string for a generic object."""
        obj_string = '#VRML_OBJ R2018a utf8\n'
        if sim_object.def_name:
            obj_string += 'DEF ' + sim_object.def_name + ' '
        obj_string += sim_object.object_name + ' {\n'
        for (param_name, param_val) in sim_object.object_parameters:
            obj_string += '  ' + param_name + ' ' + param_val + '\n'
        obj_string += "}"
        return obj_string

    def get_sensor_string(self, sensor, tab_size):
        """Generates the part of the sensor string which contains sensor type and parameters."""
        tab_str = " " * tab_size
        sensor_str = tab_str + sensor.sensor_type + " {\n"
        for sensor_field in sensor.sensor_fields:
            sensor_str = sensor_str + tab_str + "  " + sensor_field.field_name + " " + \
                         sensor_field.field_val + "\n"
        sensor_str = sensor_str + tab_str + "}\n"
        return sensor_str

    def generate_sensor_string(self, sensor_array, sensor_location):
        """Generates the Webots string for a sensor."""
        # Add text for Left Sensors if there is any
        sensor_count = 0
        sensor_string = ""
        for sensor in sensor_array:
            if sensor.sensor_location == sensor_location:
                if sensor_count == 0:
                    if sensor_location == WebotsSensor.LEFT:
                        sensor_string += "  sensorsSlotLeft [\n"
                    elif sensor_location == WebotsSensor.RIGHT:
                        sensor_string += "  sensorsSlotRight [\n"
                    elif sensor_location == WebotsSensor.LEFT_FRONT:
                        sensor_string += "  sensorsSlotLeftFront [\n"
                    elif sensor_location == WebotsSensor.LEFT_REAR:
                        sensor_string += "  sensorsSlotLeftRear [\n"
                    elif sensor_location == WebotsSensor.RIGHT_FRONT:
                        sensor_string += "  sensorsSlotRightFront [\n"
                    elif sensor_location == WebotsSensor.RIGHT_REAR:
                        sensor_string += "  sensorsSlotRightRear [\n"
                    elif sensor_location == WebotsSensor.CENTER:
                        sensor_string += "  sensorsSlotCenter [\n"
                    elif sensor_location == WebotsSensor.TOP:
                        sensor_string += "  sensorsSlotTop [\n"
                    elif sensor_location == WebotsSensor.REAR:
                        sensor_string += "  sensorsSlotRear [\n"
                    elif sensor_location == WebotsSensor.FRONT:
                        sensor_string += "  sensorsSlotFront [\n"
                    elif sensor_location == WebotsSensor.LEFT:
                        sensor_string += "  sensorsSlotUnknown [\n"
                sensor_count += 1
                sensor_string += self.get_sensor_string(sensor, 4)
        if sensor_count > 0:
            sensor_string += "  ]\n"
        return sensor_string
