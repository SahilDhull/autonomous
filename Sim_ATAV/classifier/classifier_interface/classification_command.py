"""Defines the ClassificationCommand class.
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------

"""


class ClassificationCommand(object):
    """Contains the Commands used in classification interface."""

    CMD_START_SHARED_MEMORY = 1
    CMD_CLOSE_SHARED_MEMORY = 2
    CMD_END_COMMUNICATION = 3
    CMD_CLASSIFY = 100
    CMD_CLASSIFICATION_RESULT = 110
    ACK = 250
    NACK = 251
