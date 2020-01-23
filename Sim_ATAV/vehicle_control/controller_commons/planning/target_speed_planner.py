"""Defines TargetSpeedPlanner class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import math


class TargetSpeedData(object):
    """Target speed can start at a time and last for a time defined by length. Length can be infinite.
    Target speed can alternatively start at the beginning of a segment. start_segment defines the index of the segment.
    When we set it by segment instead of time, when length is positive, it still defines how long the target speed will
    be active starting from when it is activated, when length is 0, it means the target speed is valid only for that
    segment, when length is negative (-1), it means the target speed will always be active.
    However, if a new time or segment activates a new target speed, the new one will stay override old ones until its
    length is reached."""
    def __init__(self, event_type, start_time=0.0, start_segment=-1, length=math.inf, target_speed=0.0):
        self.event_type = event_type
        self.start_time = start_time
        self.start_segment = start_segment
        self.length = length
        self.target_speed = target_speed


class TargetSpeedPlanner(object):
    def __init__(self, default_speed=0.0):
        self.target_speed_stack = []
        self.target_speed_stack.append(
            TargetSpeedData(event_type='time', start_time=0.0, length=math.inf, target_speed=default_speed))
        self.time_based_targets_dict = {}
        self.segment_based_targets_dict = {}

    def add_target_speed_data(self, target_speed_data):
        if target_speed_data.event_type == 'time':
            self.time_based_targets_dict[target_speed_data.start_time] = target_speed_data
        else:
            self.segment_based_targets_dict[target_speed_data.start_segment] = target_speed_data

    def is_top_item_old(self, cur_time_ms, cur_segment_ind=-1):
        is_old = False
        if self.target_speed_stack:
            top = self.target_speed_stack[-1]
            if top.event_type == 'time' and cur_time_ms > top.start_time + top.length:
                is_old = True
            elif top.event_type == 'segment':
                if top.length == 0 and cur_segment_ind > top.start_segment:  # Length is 1 segment
                    is_old = True
                elif top.length > 0 and cur_time_ms > top.start_time + top.length:
                    is_old = True
        return is_old

    def clear_old_from_stack(self, cur_time_ms, cur_segment_ind=-1):
        while self.is_top_item_old(cur_time_ms, cur_segment_ind):
            self.target_speed_stack.pop()

    def add_new_to_stack(self, cur_time_ms, cur_segment_ind=-1):
        if len(self.time_based_targets_dict.keys()) > 0:
            min_time_key = min(self.time_based_targets_dict.keys())
            if self.time_based_targets_dict[min_time_key].start_time <= cur_time_ms:
                self.target_speed_stack.append(self.time_based_targets_dict[min_time_key])
                del self.time_based_targets_dict[min_time_key]
        if len(self.segment_based_targets_dict.keys()) > 0:
            min_segment_key = min(self.segment_based_targets_dict.keys())
            if cur_segment_ind >= self.segment_based_targets_dict[min_segment_key].start_segment:
                self.segment_based_targets_dict[min_segment_key].start_time = cur_time_ms
                self.target_speed_stack.append(self.segment_based_targets_dict[min_segment_key])
                del self.segment_based_targets_dict[min_segment_key]

    def get_current_target_speed(self, cur_time_ms, cur_segment_ind=-1):
        self.clear_old_from_stack(cur_time_ms, cur_segment_ind)
        self.add_new_to_stack(cur_time_ms, cur_segment_ind)
        return self.target_speed_stack[-1].target_speed
