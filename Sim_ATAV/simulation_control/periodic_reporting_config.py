"""Defines PeriodicReportingConfig class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""


class PeriodicReportingConfig(object):
    """PeriodicReportingConfig is used for the supervisor to periodically broadcast data from simulation.
    This data is used as the ground truth information etc."""
    REPORT_AT_EVERY_STEP = 0
    REPORT_ONLY_ONCE = -1
    DO_NOT_REPORT = -2

    ALL_IDS = 0

    def __init__(self, item_type, item_id, report_type, reporting_period):
        self.item_type = item_type  # as string
        self.item_id = item_id  # Can use ALL_IDS for all items of that type
        self.report_type = report_type  # as string
        self.reporting_period = reporting_period  # Can use REPORT_ONLY_ONCE for non-periodic, one-time reporting.
