"""Defines WebotsFog Class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


class WebotsFog(object):
    """User Configurable Fog Structure to use in Webots environment"""
    def __init__(self):
        self.def_name = "FOG"
        self.fog_type = "LINEAR"
        self.color = [0.93, 0.96, 1.0]
        self.visibility_range = 1000.0
