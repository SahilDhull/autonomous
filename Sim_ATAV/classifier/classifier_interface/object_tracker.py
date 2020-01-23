"""Defines ObjectTracker Class
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""

import cv2


class ObjectTracker(object):
    """A class which uses OpenCV to do object tracking in images."""
    def __init__(self):
        self.trackers = []
        self.tracked_object_boxes = []

    def set_tracked_objects(self, object_boxes, image):
        """Set known bounding boxes of the objects."""
        self.tracked_object_boxes = []
        del self.trackers
        self.trackers = []
        if len(image) > 0:
            for bbox_arr in object_boxes:
                # Opencv accepts bounding boxes as tuples not arrays
                bbox = (bbox_arr[0], bbox_arr[1], bbox_arr[2], bbox_arr[3])
                self.tracked_object_boxes.append(bbox)
                self.trackers.append(cv2.TrackerMedianFlow_create())
                tracker = self.trackers[-1]
                tracker.init(image, bbox)

    def get_tracking_results(self, image):
        """Update and return new bounding boxes of the objects, together with the old positions."""
        objects_old_position = self.tracked_object_boxes[:]
        if len(image) > 0:
            for (ind, tracker) in enumerate(self.trackers):
                (success, bbox) = tracker.update(image)
                # Opencv returns bounding boxes as tuples not arrays
                bbox_arr = [bbox[0], bbox[1], bbox[2], bbox[3]]
                if success:
                    self.tracked_object_boxes[ind] = bbox_arr[:]
        return self.tracked_object_boxes, objects_old_position
