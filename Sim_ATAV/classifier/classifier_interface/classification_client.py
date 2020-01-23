"""Defines the ClassificationClient class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------

"""

import time
import glob
import cv2
import numpy as np
from Sim_ATAV.classifier.classifier_interface.classification_client_interface import ClassificationClientInterface


class ClassificationClient(object):
    """ClassificationClient connects to classification server for object detection."""

    CAR_LABEL = 0
    PEDESTRIAN_LABEL = 1

    def __init__(self, is_debug_mode=False):
        # Constants and configuration
        self.interface = ClassificationClientInterface(is_debug_mode=is_debug_mode)

    def establish_communication(self, file_name=None, file_size=None):
        """Creates a shared memory, connects to the server and
        sends the shared memory information to the server."""
        self.interface.connect_to_classification_server()
        self.interface.setup_shared_memory(file_name=file_name, file_size=file_size)

    def get_classification_results(self, data, width, height):
        """Asks server to classify the image in the shared memory and
        gets the image classification results."""
        return self.interface.classify_data(data, width, height)

    def close_communication(self):
        """Closes the shared memory file and the connection with the server."""
        self.interface.close_shared_memory()
        self.interface.end_communication_with_server()


def main():
    """This is for running the classification client standalone."""
    shared_file_name = "Local\\WebotsCameraImage"
    shared_file_size = 1437696
    classification_client = ClassificationClient()
    classification_client.establish_communication(file_name=shared_file_name,
                                                  file_size=shared_file_size)
    start_time = time.time()
    for image_file in glob.iglob('./data/webots3.png'):
        image = cv2.imread(image_file)
        (width, height, depth) = image.shape
        image = np.ravel(image)
        classification_client.get_classification_results(image, width, height)
    classification_client.close_communication()
    elapsed_time = time.time() - start_time
    print("Elapsed time: {}".format(elapsed_time))


if __name__ == "__main__":
    main()
