"""Defines GroundTruthGenerator class.
----------------------------------------------------------------------------------------------------------
This file is part of Sim-ATAV project and licensed under MIT license.
Copyright (c) 2018 Cumhur Erkan Tuncali, Georgios Fainekos, Danil Prokhorov, Hisahiro Ito, James Kapinski.
For questions please contact:
C. Erkan Tuncali (etuncali [at] asu.edu)
----------------------------------------------------------------------------------------------------------
"""
import numpy as np
from Sim_ATAV.classifier.classifier_interface.camera_projection import CameraProjection


class GroundTruthGenerator(object):
    """GroundTruthGenerator class handles the generation of detection boxes and other ground truth
    information from the given camera, object position and rotation information."""
    def __init__(self):
        self.camera_projection = CameraProjection()
        self.cam_position = [0.0, 0.0, 0.0]
        self.cam_rotation = np.eye(3)

    def set_camera_parameters(self, camera_width, camera_height, camera_hor_fov):
        """Set camera details."""
        self.camera_projection.set_camera_specific_settings(camera_width, camera_height, camera_hor_fov)

    def compute_camera_position_rotation(self, vhc_position, vhc_rotation, camera_local_position, camera_local_rotation):
        cam_local_coord = np.array(camera_local_position)
        cam_local_coord.shape = (3, 1)
        cam_position = self.camera_projection.convert_from_local_to_world_coordinates(vhc_rotation,
                                                                                      vhc_position,
                                                                                      cam_local_coord)
        cam_rot = np.array(vhc_rotation)
        cam_rot.shape = (3, 3)
        cam_rotation = np.matmul(camera_local_rotation, cam_rot)
        return cam_position, cam_rotation

    def get_object_detection_box(self, obj_position, obj_rotation, obj_corners, cam_position, cam_rotation):
        """Get the corners of the detection box."""
        obj_pos = np.array(obj_position)
        obj_pos.shape = (3, 1)
        obj_rot = np.array(obj_rotation)
        obj_rot.shape = (3, 3)

        min_x = self.camera_projection.cam_width
        max_x = 0
        min_y = self.camera_projection.cam_height
        max_y = 0
        for corner_id in obj_corners:
            local_coord = np.array(obj_corners[corner_id])
            local_coord.shape = (3, 1)
            world_coord = self.camera_projection.convert_from_local_to_world_coordinates(obj_rot,
                                                                                         obj_pos,
                                                                                         local_coord)
            cam_coord = self.camera_projection.convert_from_world_to_camera_coordinates(cam_rotation,
                                                                                        cam_position,
                                                                                        world_coord)
            image_coord = self.camera_projection.convert_from_camera_to_image_coordinates(cam_coord)
            if image_coord[2] > 0.001:
                x_pos = int(image_coord[0] / image_coord[2])
                y_pos = int(image_coord[1] / image_coord[2])
                if x_pos < min_x:
                    min_x = x_pos
                if x_pos > max_x:
                    max_x = x_pos
                if y_pos < min_y:
                    min_y = y_pos
                if y_pos > max_y:
                    max_y = y_pos
        min_x_updated = max(0, min_x)
        min_y_updated = max(0, min_y)
        max_x_updated = min(self.camera_projection.cam_width, max_x)
        max_y_updated = min(self.camera_projection.cam_height, max_y)
        if max_x - min_x > 0 and max_y - min_y > 0:
            area_org = (max_x - min_x) * (max_y - min_y)
            area_updated = (max_x_updated - min_x_updated) * (max_y_updated - min_y_updated)
            truncation = abs((area_org - area_updated) / area_org)
        else:
            truncation = 1.0  #completely outside the image window, hence completely truncated

        return min_x_updated, min_y_updated, max_x_updated, max_y_updated, truncation

    def get_all_vhc_and_ped_detection_boxes(self,
                                            self_vhc_id,
                                            cam_local_position,
                                            cam_local_rotation,
                                            vhc_pos_dict,
                                            vhc_rot_dict,
                                            vhc_corners_dict,
                                            ped_pos_dict,
                                            ped_rot_dict,
                                            ped_corners_dict):
        vhc_position = np.array(vhc_pos_dict[self_vhc_id])
        vhc_position.shape = (3, 1)
        vhc_rotation = np.array(vhc_rot_dict[self_vhc_id])
        vhc_rotation.shape = (3, 3)
        (cam_position, cam_rotation) = \
            self.compute_camera_position_rotation(vhc_position, vhc_rotation, cam_local_position, cam_local_rotation)
        detection_box_list = []
        for obj_id in vhc_pos_dict:
            if obj_id != self_vhc_id and (obj_id in vhc_rot_dict and obj_id in vhc_corners_dict):  # Defensive check
                obj_class_name = 'Car'
                (x_pos_min, y_pos_min, x_pos_max, y_pos_max, truncation) = \
                    self.get_object_detection_box(vhc_pos_dict[obj_id],
                                                  vhc_rot_dict[obj_id],
                                                  vhc_corners_dict[obj_id],
                                                  cam_position,
                                                  cam_rotation)
                if truncation < 0.85:
                    detection_box_list.append((obj_class_name, obj_id, truncation, x_pos_min, y_pos_min, x_pos_max, y_pos_max))

        for obj_id in ped_pos_dict:
            if obj_id in ped_rot_dict and obj_id in ped_corners_dict:  # Defensive check
                obj_class_name = 'Pedestrian'
                (x_pos_min, y_pos_min, x_pos_max, y_pos_max, truncation) = \
                    self.get_object_detection_box(ped_pos_dict[obj_id],
                                                  ped_rot_dict[obj_id],
                                                  ped_corners_dict[obj_id],
                                                  cam_position,
                                                  cam_rotation)
                if truncation < 0.85:
                    detection_box_list.append((obj_class_name, obj_id, truncation, x_pos_min, y_pos_min, x_pos_max, y_pos_max))
        return detection_box_list

    def get_single_obj_bounding_box(self,
                                    self_vhc_id,
                                    requested_obj_id,
                                    requested_obj_type,
                                    cam_local_position,
                                    cam_local_rotation,
                                    vhc_pos_dict,
                                    vhc_rot_dict,
                                    vhc_corners_dict,
                                    ped_pos_dict,
                                    ped_rot_dict,
                                    ped_corners_dict):
        """Returns the ground truth bounding box for an object."""
        x_pos_min = 0
        x_pos_max = 0
        y_pos_max = 0
        y_pos_min = 0
        truncation = 1.0
        obj_class_name = 'N/A'

        if self_vhc_id in vhc_pos_dict:
            vhc_position = np.array(vhc_pos_dict[self_vhc_id])
            vhc_position.shape = (3, 1)
            vhc_rotation = np.array(vhc_rot_dict[self_vhc_id])
            vhc_rotation.shape = (3, 3)
            (cam_position, cam_rotation) = \
                self.compute_camera_position_rotation(vhc_position, vhc_rotation, cam_local_position, cam_local_rotation)
            if requested_obj_type in ['pedestrian', 'Pedestrian', 'ped', 'Ped'] and requested_obj_id in ped_pos_dict:
                obj_class_name = 'Pedestrian'
                (x_pos_min, y_pos_min, x_pos_max, y_pos_max, truncation) = \
                    self.get_object_detection_box(ped_pos_dict[requested_obj_id],
                                                  ped_rot_dict[requested_obj_id],
                                                  ped_corners_dict[requested_obj_id],
                                                  cam_position,
                                                  cam_rotation)
            elif requested_obj_type in ['car', 'Car', 'vhc', 'Vhc', 'vehicle', 'Vehicle'] and requested_obj_id in vhc_pos_dict:
                obj_class_name = 'Car'
                (x_pos_min, y_pos_min, x_pos_max, y_pos_max, truncation) = \
                    self.get_object_detection_box(vhc_pos_dict[requested_obj_id],
                                                  vhc_rot_dict[requested_obj_id],
                                                  vhc_corners_dict[requested_obj_id],
                                                  cam_position,
                                                  cam_rotation)
        bbox_x = (x_pos_min + x_pos_max) / 2.0
        bbox_y = (y_pos_min + y_pos_max) / 2.0
        bbox_w = x_pos_max - x_pos_min
        bbox_h = y_pos_max - y_pos_min
        return obj_class_name, truncation, [bbox_x, bbox_y, bbox_w, bbox_h]

    def get_all_obj_bounding_boxes(self,
                                   self_vhc_id,
                                   cam_local_position,
                                   cam_local_rotation,
                                   vhc_pos_dict,
                                   vhc_rot_dict,
                                   vhc_corners_dict,
                                   ped_pos_dict,
                                   ped_rot_dict,
                                   ped_corners_dict):
        bbox_dict = {}

        if self_vhc_id in vhc_pos_dict:
            vhc_position = np.array(vhc_pos_dict[self_vhc_id])
            vhc_position.shape = (3, 1)
            vhc_rotation = np.array(vhc_rot_dict[self_vhc_id])
            vhc_rotation.shape = (3, 3)
            (cam_position, cam_rotation) = \
                self.compute_camera_position_rotation(vhc_position, vhc_rotation, cam_local_position, cam_local_rotation)
            for obj_id in ped_pos_dict:
                if obj_id in ped_rot_dict and obj_id in ped_corners_dict:  # Defensive check
                    obj_class_name = 'Pedestrian'
                    (x_pos_min, y_pos_min, x_pos_max, y_pos_max, truncation) = \
                        self.get_object_detection_box(ped_pos_dict[obj_id],
                                                      ped_rot_dict[obj_id],
                                                      ped_corners_dict[obj_id],
                                                      cam_position,
                                                      cam_rotation)
                    bbox_x = (x_pos_min + x_pos_max) / 2.0
                    bbox_y = (y_pos_min + y_pos_max) / 2.0
                    bbox_w = x_pos_max - x_pos_min
                    bbox_h = y_pos_max - y_pos_min
                    bbox_dict[(obj_id, obj_class_name)] = (truncation, [bbox_x, bbox_y, bbox_w, bbox_h])
            for obj_id in vhc_pos_dict:
                if obj_id != self_vhc_id and (obj_id in vhc_rot_dict and obj_id in vhc_corners_dict):  # Defensive check
                    obj_class_name = 'Car'
                    (x_pos_min, y_pos_min, x_pos_max, y_pos_max, truncation) = \
                        self.get_object_detection_box(vhc_pos_dict[obj_id],
                                                      vhc_rot_dict[obj_id],
                                                      vhc_corners_dict[obj_id],
                                                      cam_position,
                                                      cam_rotation)
                    bbox_x = (x_pos_min + x_pos_max) / 2.0
                    bbox_y = (y_pos_min + y_pos_max) / 2.0
                    bbox_w = x_pos_max - x_pos_min
                    bbox_h = y_pos_max - y_pos_min
                    bbox_dict[(obj_id, obj_class_name)] = (truncation, [bbox_x, bbox_y, bbox_w, bbox_h])
        return bbox_dict

    def compute_bounding_box(self,
                             self_pos,
                             self_rot,
                             cam_local_position,
                             cam_local_rotation,
                             obj_pos,
                             obj_rot,
                             obj_corners):
        vhc_position = self_pos
        vhc_position.shape = (3, 1)
        vhc_rotation = self_rot
        vhc_rotation.shape = (3, 3)
        (cam_position, cam_rotation) = \
            self.compute_camera_position_rotation(vhc_position, vhc_rotation, cam_local_position, cam_local_rotation)

        (x_pos_min, y_pos_min, x_pos_max, y_pos_max, _truncation) = \
            self.get_object_detection_box(obj_pos,
                                          obj_rot,
                                          obj_corners,
                                          cam_position,
                                          cam_rotation)
        bbox_x = (x_pos_min + x_pos_max) / 2.0
        bbox_y = (y_pos_min + y_pos_max) / 2.0
        bbox_w = x_pos_max - x_pos_min
        bbox_h = y_pos_max - y_pos_min
        return [bbox_x, bbox_y, bbox_w, bbox_h]

    def check_occlusions(self, detection_box_list, vhc_pos_dict, ped_pos_dict, cam_position):
        """Return an array of occlusion values for the objects with detection_boxes."""
        occlusion_list = [0]*len(detection_box_list)
        for (obj_ind, obj_detection_box) in enumerate(detection_box_list[:]):
            for (temp_other_ind, other_detection_box) in enumerate(detection_box_list[obj_ind + 1:]):
                # Check intersection:
                other_ind = obj_ind + 1 + temp_other_ind  # index starts from 0 in the second enumerate
                left = max(obj_detection_box[3], other_detection_box[3])  # max of min_x's
                right = min(obj_detection_box[5], other_detection_box[5])  # min of max_x's
                top = max(obj_detection_box[4], other_detection_box[4])  # max of min_y's
                bottom = min(obj_detection_box[6], other_detection_box[6])  # min of max_y's
                if left < right and top < bottom:
                    # Intersects. Find which one is away
                    obj_id = obj_detection_box[1]
                    obj_class = obj_detection_box[0]
                    if obj_class == 'Car':
                        obj_pos = np.array(vhc_pos_dict[obj_id])
                    else:
                        obj_pos = np.array(ped_pos_dict[obj_id])
                    obj_pos.shape = (3, 1)
                    obj_dist_to_cam = np.linalg.norm(obj_pos - cam_position)

                    other_id = other_detection_box[1]
                    other_class = other_detection_box[0]
                    if other_class == 'Car':
                        other_pos = np.array(vhc_pos_dict[other_id])
                    else:
                        other_pos = np.array(ped_pos_dict[other_id])
                    other_pos.shape = (3, 1)
                    other_dist_to_cam = np.linalg.norm(other_pos - cam_position)

                    if obj_dist_to_cam < other_dist_to_cam:
                        occ_ind = other_ind
                        occluded_obj_area = (other_detection_box[5] - other_detection_box[3]) * \
                                            (other_detection_box[6] - other_detection_box[4])
                    else:
                        occ_ind = obj_ind
                        occluded_obj_area = (obj_detection_box[5] - obj_detection_box[3]) * \
                                            (obj_detection_box[6] - obj_detection_box[4])
                    intersection_area = (right - left) * (bottom - top)
                    if intersection_area / occluded_obj_area > 0.5:
                        occlusion_type = 2
                    elif intersection_area / occluded_obj_area > 0.1:
                        occlusion_type = 1
                    else:
                        occlusion_type = 0
                    occlusion_list[occ_ind] = max(occlusion_list[occ_ind], occlusion_type)
        return occlusion_list

    def get_all_vhc_and_ped_ground_truth_info(self,
                                              self_vhc_id,
                                              cam_local_position,
                                              cam_local_rotation,
                                              vhc_pos_dict,
                                              vhc_rot_dict,
                                              vhc_corners_dict,
                                              ped_pos_dict,
                                              ped_rot_dict,
                                              ped_corners_dict):
        vhc_position = np.array(vhc_pos_dict[self_vhc_id])
        vhc_position.shape = (3, 1)
        vhc_rotation = np.array(vhc_rot_dict[self_vhc_id])
        vhc_rotation.shape = (3, 3)
        (cam_position, cam_rotation) = \
            self.compute_camera_position_rotation(vhc_position, vhc_rotation, cam_local_position, cam_local_rotation)
        detection_box_list = self.get_all_vhc_and_ped_detection_boxes(self_vhc_id,
                                                                      cam_local_position,
                                                                      cam_local_rotation,
                                                                      vhc_pos_dict,
                                                                      vhc_rot_dict,
                                                                      vhc_corners_dict,
                                                                      ped_pos_dict,
                                                                      ped_rot_dict,
                                                                      ped_corners_dict)
        occlusion_list = self.check_occlusions(detection_box_list, vhc_pos_dict, ped_pos_dict, cam_position)
        ground_truth_list = []
        for (obj_ind, detection_box) in enumerate(detection_box_list):
            # KITTI Dataset has the following labels:
            # [type, truncation, occluded, alpha observation angle,
            # left, top, right, bottom, height (m), width (m), length (m),
            # camera coord x (m), camera coord y (m), camera coord z (m), rotation]
            #
            # SqueezeDet ignores alpha observation angle, height (m), width (m), length (m),
            # camera coord x (m), camera coord y (m), camera coord z (m), rotation
            ground_truth_list.append((detection_box[0],
                                      detection_box[2],
                                      occlusion_list[obj_ind],
                                      0.0,
                                      detection_box[3],
                                      detection_box[4],
                                      detection_box[5],
                                      detection_box[6],
                                      0.0,
                                      0.0,
                                      0.0,
                                      0.0,
                                      0.0,
                                      0.0,
                                      0.0))
        return ground_truth_list

    def write_all_vhc_and_ped_ground_truth_info_to_file(self,
                                                        file_name,
                                                        self_vhc_id,
                                                        cam_local_position,
                                                        cam_local_rotation,
                                                        vhc_pos_dict,
                                                        vhc_rot_dict,
                                                        vhc_corners_dict,
                                                        ped_pos_dict,
                                                        ped_rot_dict,
                                                        ped_corners_dict):
        """Generate and record ground truth labels into file"""
        with open(file_name, 'w') as f:
            file_str = ''
            ground_truth_list = self.get_all_vhc_and_ped_ground_truth_info(self_vhc_id,
                                                                           cam_local_position,
                                                                           cam_local_rotation,
                                                                           vhc_pos_dict,
                                                                           vhc_rot_dict,
                                                                           vhc_corners_dict,
                                                                           ped_pos_dict,
                                                                           ped_rot_dict,
                                                                           ped_corners_dict)
            for ground_truth in ground_truth_list:
                file_str += ground_truth[0]
                file_str += ' '
                file_str += '{0:.2f}'.format(float(ground_truth[1]))
                file_str += ' '
                file_str += str(int(ground_truth[2]))
                for ind in range(3, 15):
                    file_str += ' '
                    file_str += '{0:.2f}'.format(float(ground_truth[ind]))
                file_str += '\n'
            f.write(file_str)
            f.close()
