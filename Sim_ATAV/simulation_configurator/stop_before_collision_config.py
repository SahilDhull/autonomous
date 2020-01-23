"""Defines StopBeforeCollisionConfig class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


class StopBeforeCollisionConfig(object):
    """We can configure objects in the simulation to stop magically before colliding into other objects.
    This is sometimes needed because those collisions are out of our testing focus."""
    def __init__(self, item_to_stop_type, item_to_stop_ind, item_not_to_collide_type, item_not_to_collide_ind):
        self.item_to_stop_type = item_to_stop_type
        self.item_to_stop_ind = item_to_stop_ind
        self.item_not_to_collide_type = item_not_to_collide_type
        self.item_not_to_collide_ind = item_not_to_collide_ind
