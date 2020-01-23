"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
class DetectionEvaluationConfig(object):
    def __init__(self, vehicle_id=0, sensor_type='overall', sensor_id=0,
                 target_objs=None, eval_type='localization', eval_alg=None):
        self.vehicle_id = vehicle_id
        self.sensor_type = sensor_type
        self.sensor_id = sensor_id
        if target_objs is None:
            self.target_objs = []
        else:
            self.target_objs = target_objs
        self.eval_type = eval_type
        self.eval_alg = eval_alg

    def add_target_object(self, object_type, object_id):
        self.target_objs.append((object_type, object_id))

    def get_target_obj_info_as_dictionary_key(self, target_obj_ind):
        """When we use detection evaluation config in the simulation trace, this is how we use it as a key in the
        trajectory dictionary for easy reference."""
        return (self.vehicle_id, self.sensor_type, self.sensor_id, self.eval_type, self.eval_alg,
                self.target_objs[target_obj_ind][0], self.target_objs[target_obj_ind][1])
