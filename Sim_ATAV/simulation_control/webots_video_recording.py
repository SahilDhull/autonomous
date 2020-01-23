"""Defines WebotsSimObject Class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


class WebotsVideoRecording(object):
    """Structure to describe video recording settings in Webots environment"""
    def __init__(self):
        self.width = 1920
        self.height = 1080
        self.codec = 1 # ignored
        self.quality = 100
        self.acceleration = 1
        self.is_caption = False
        self.caption_name = 'SimATAV'
        self.filename = 'SimATAV_video.mp4'
