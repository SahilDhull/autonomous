"""
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import sys


class ConsoleOutput(object):
    def __init__(self, debug_mode):
        self.debug_mode = debug_mode

    def debug_print(self, print_str):
        if self.debug_mode:
            print(print_str)
            sys.stdout.flush()
