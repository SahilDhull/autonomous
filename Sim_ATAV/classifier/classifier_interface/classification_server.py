"""Defines the ClassificationServer class.
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------

"""

from Sim_ATAV.classifier.classifier_interface.classification_server_interface import ClassificationServerInterface
from Sim_ATAV.classifier.classifier_interface.classifier import Classifier


class ClassificationServer(object):
    """ClassificationServer serves to perform image classification and object detection."""

    def __init__(self, port=10101, is_debug_mode=False, classification_engine=None):
        # Constants and configuration
        if classification_engine is not None:
            self.interface = \
                ClassificationServerInterface(port=port,
                                              is_debug_mode=is_debug_mode,
                                              classification_engine=classification_engine)
        else:
            self.interface = None

    def run_server(self):
        """Starts the classification server."""
        while True and self.interface is not None:
            print("Server Starting!")
            self.interface.setup_connection()
            self.interface.start_service()


def run_classification_server():
    """Run classification server standalone."""
    classification_engine = Classifier()
    classification_server = ClassificationServer(port=10101,
                                                 is_debug_mode=False,
                                                 classification_engine=classification_engine)
    classification_server.run_server()


def main():
    """Run classification server standalone."""
    classification_engine = Classifier()
    classification_server = ClassificationServer(port=10101,
                                                 is_debug_mode=False,
                                                 classification_engine=classification_engine)
    classification_server.run_server()


if __name__ == "__main__":
    main()
