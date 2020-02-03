"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
from Sim_ATAV.common.controller_communication_interface import ControllerCommunicationInterface
from Sim_ATAV.simulation_control.item_description import ItemDescription
from Sim_ATAV.vehicle_control.controller_commons.perf_evaluation.detection_performance_monitor \
    import DetectionPerformanceMonitor
from Sim_ATAV.vehicle_control.controller_commons.perf_evaluation.visibility_monitor import VisibilityMonitor


class CommunicationModule(object):
    def __init__(self, controller=None):
        self.controller = controller

    def receive_and_update(self, cur_time_ms):
        if (self.controller is not None and
                self.controller.receiver is not None and
                self.controller.controller_comm_interface is not None):
            messages = self.controller.controller_comm_interface.receive_all_communication(self.controller.receiver)
            command_list = self.controller.controller_comm_interface.extract_all_commands_from_message(messages)
            if self.controller.ground_truth_detector is not None:
                self.controller.ground_truth_detector.update_detections(command_list, cur_time_ms)

            for command_item in command_list:
                command = command_item[0]
                if command == ControllerCommunicationInterface.SET_CONTROLLER_PARAMETERS_MESSAGE:
                    parameter = command_item[1]
                    if parameter.get_vehicle_id() == self.controller.self_vhc_id:
                        if parameter.get_parameter_name() == 'target_position':
                            parameter_data = parameter.get_parameter_data()
                            if self.controller.path_planner is not None:
                                self.controller.path_planner.add_waypoint(parameter_data)
                elif command == ControllerCommunicationInterface.SET_DETECTION_MONITOR:
                    detection_eval_config = command_item[1]
                    if detection_eval_config.vehicle_id == self.controller.self_vhc_id:
                        if self.controller.detection_perf_monitor is None:
                            self.controller.detection_perf_monitor = \
                                DetectionPerformanceMonitor(ground_truth_detector=self.controller.ground_truth_detector)
                            self.controller.detection_perf_monitor.set_perception_system(
                                self.controller.perception_system)
                        self.controller.detection_perf_monitor.add_monitor(detection_eval_config)
                elif command == ControllerCommunicationInterface.SET_VISIBILITY_MONITOR:
                    visibility_eval_config = command_item[1]
                    if visibility_eval_config.vehicle_id == self.controller.self_vhc_id:
                        if self.controller.visibility_monitor is None:
                            self.controller.visibility_monitor = \
                                VisibilityMonitor(ground_truth_detector=self.controller.ground_truth_detector)
                        self.controller.visibility_monitor.add_monitor(visibility_eval_config)
            if self.controller.path_planner is not None:
                self.controller.path_planner.apply_path_changes()
            # print(self.controller.path_planner.path_following_tools.target_path)
            # print(self.controller.path_planner.path_following_tools.path_details)

    def transmit_control_data(self, control_throttle, control_steering):
        if self.controller.emitter is not None:
            message = self.controller.controller_comm_interface.generate_control_action_message(
                self.controller.self_vhc_id,
                ItemDescription.VEHICLE_CONTROL_THROTTLE,
                control_throttle)
            self.controller.emitter.send(message)
            message = self.controller.controller_comm_interface.generate_control_action_message(
                self.controller.self_vhc_id,
                ItemDescription.VEHICLE_CONTROL_STEERING,
                control_steering)
            self.controller.emitter.send(message)

    def transmit_detection_evaluation_data(self):
        if self.controller.emitter is not None and self.controller.detection_perf_monitor is not None:
            detection_evaluations = self.controller.detection_perf_monitor.get_evaluations()
            for (idx, detection_evaluation) in enumerate(detection_evaluations):
                message = self.controller.controller_comm_interface.generate_detection_evaluation_message(
                    idx=idx,value=detection_evaluation)
                self.controller.emitter.send(message)

    def transmit_visibility_evaluation_data(self):
        if self.controller.emitter is not None and self.controller.visibility_monitor is not None:
            visibility_evaluations = self.controller.visibility_monitor.get_evaluations()
            for (idx, visibility_evaluation) in enumerate(visibility_evaluations):
                message = self.controller.controller_comm_interface.generate_visibility_evaluation_message(
                    idx=idx,value=visibility_evaluation)
                self.controller.emitter.send(message)
