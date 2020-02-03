"""Modified from SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import timeit
import cv2
import numpy as np
import tensorflow as tf
from Sim_ATAV.external.squeezeDet.squeezeDet.train import _draw_box
from Sim_ATAV.external.squeezeDet.squeezeDet.config import *
from Sim_ATAV.external.squeezeDet.squeezeDet.nets import *
FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class Classifier(object):
    """Classifier class is a wrapper around SqueezeDet.
    Provides the methods to do the object detection on a pretrained model."""
    DET_BOX_X_IND = 0
    DET_BOX_Y_IND = 1
    DET_BOX_WIDTH_IND = 2
    DET_BOX_HEIGHT_IND = 3

    CAR_BOX_COLOR = (255, 0, 0)
    CYCLIST_BOX_COLOR = (0, 255, 0)
    PEDESTRIAN_BOX_COLOR = (0, 0, 255)

    CAR_BOX_COLOR_HEX = 0xFF0000
    CYCLIST_BOX_COLOR_HEX = 0x00FF00
    PEDESTRIAN_BOX_COLOR_HEX = 0x0000FF

    CAR_CLASS_LABEL = 0
    PEDESTRIAN_CLASS_LABEL = 1
    CYCLIST_CLASS_LABEL = 2

    PED_DISTANCE_SCALE_FACTOR = 2793  # Focal length * known height for pedestrian
    CAR_DISTANCE_SCALE_FACTOR = 2483  # Focal length * known height for car

    KNOWN_PED_HEIGHT = 1.8
    KNOWN_CAR_HEIGHT = 1.6

    # Pre-trained models in SqueezeDet:
    SQUEEZE_DET = 'squeezeDet'
    SQUEEZE_DET_PLUS = 'squeezeDet+'

    SQUEEZE_DET_PLUS_SAVED_MODEL_FILE = FILE_PATH + '/data/model_checkpoints/squeezeDetPlus/model.ckpt-95000'
    SQUEEZE_DET_SAVED_MODEL_FILE = FILE_PATH + '/data/model_checkpoints/squeezeDet/model.ckpt-200500'

    IMAGE_WINDOW_NAME = "squeezeDet Image"
    PROBABILITY_THRESHOLD = 0.5

    def __init__(self, is_gpu=True, processor_id=0, is_show_image=False):
        self.model_type = self.SQUEEZE_DET
        self.is_gpu = is_gpu
        self.processor_id = processor_id
        self.sess = None
        self.model_config = None
        self.model = None
        self.frame_id = 0
        self.is_show_image = is_show_image
        self.class_color_dict = {
            'car': self.CAR_BOX_COLOR,
            'cyclist': self.CYCLIST_BOX_COLOR,
            'pedestrian': self.PEDESTRIAN_BOX_COLOR
        }
        if self.is_show_image:
            cv2.namedWindow(self.IMAGE_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    def start_classification_engine(self):
        """Load the SqueezeDet model and start a tensorFlow session."""
        with tf.Graph().as_default():
            # Load model
            if self.model_type == self.SQUEEZE_DET:
                self.model_config = kitti_squeezeDet_config()
                self.model_config.BATCH_SIZE = 1
                # model parameters will be restored from checkpoint
                self.model_config.LOAD_PRETRAINED_MODEL = False
                self.model = SqueezeDet(self.model_config, is_gpu=self.is_gpu, processor_id=self.processor_id)
                self.model_config.PLOT_PROB_THRESH = self.PROBABILITY_THRESHOLD
            elif self.model_type == self.SQUEEZE_DET_PLUS:
                self.model_config = kitti_squeezeDetPlus_config()
                self.model_config.BATCH_SIZE = 1
                self.model_config.LOAD_PRETRAINED_MODEL = False
                self.model = SqueezeDetPlus(self.model_config, is_gpu=self.is_gpu, processor_id=self.processor_id)

            saver = tf.train.Saver(self.model.model_params)
            # Start a tensorFlow session:
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            # Load the pre-trained model (restore from the saved file):
            if self.model_type == self.SQUEEZE_DET:
                saver.restore(self.sess, self.SQUEEZE_DET_SAVED_MODEL_FILE)
            elif self.model_type == self.SQUEEZE_DET_PLUS:
                saver.restore(self.sess, self.SQUEEZE_DET_PLUS_SAVED_MODEL_FILE)
            # Save to file for tensorboard
            #file_writer = tf.summary.FileWriter('D:\\squeezedet_graph', self.sess.graph)

    def do_object_detection_on_raw_data(self, data, width, height, is_return_det_image=False, is_return_org_image=False):
        """Take the given 1D array of image data, convert it to an image and do object detection.
        Return detection image (if requested), boxes, probabilities and classes."""
        SAVE_TO_FILE = False
        start_time = timeit.default_timer()
        original_image = self.convert_data_to_image(data, width, height)
        elapsed_time = timeit.default_timer() - start_time
        print('Classifier convert data to image: {}'.format(elapsed_time))
        start_time = timeit.default_timer()
        if is_return_org_image:
            image = original_image.astype(np.float32, copy=True)
        else:
            # Do not copy all the image if the original image is not requested.
            image = original_image.astype(np.float32, copy=False)

        elapsed_time = timeit.default_timer() - start_time
        print('Classifier image type change: {}'.format(elapsed_time))
        if SAVE_TO_FILE:
            self.frame_id += 1
            (im_out, (final_boxes, final_probs, final_class)) = \
                self.do_object_detection(image, is_return_image=(SAVE_TO_FILE or is_return_det_image))
            self.save_output_to_file('det_output_'+\
                                     str(self.frame_id)+'.png',
                                     im_out)
        else:
            (im_out, (final_boxes, final_probs, final_class)) = \
                self.do_object_detection(image, is_return_image=(self.is_show_image or is_return_det_image))

        if self.is_show_image:
            # cv2.imwrite takes BGR values in [0, 255] range. But imshow takes them in [0, 1] range.
            # We convert images to [0, 1] range by dividing them by 255.0.
            cv2.imshow(self.IMAGE_WINDOW_NAME, im_out/float(255))
            cv2.waitKey(1)

        if not is_return_det_image:
            im_out = None

        return final_boxes, final_probs, final_class, im_out, original_image

    def do_object_detection_on_raw_data_for_matlab(self, data, width, height):
        """Take the given 1D array of image data, convert it to an image and do object detection.
        Return detection image (if requested), boxes, probabilities and classes."""
        original_image = self.convert_data_to_image(np.array(data), width, height, depth=3)
        image = original_image.astype(np.float32, copy=False)
        (_im_out, (final_boxes, final_probs, final_class)) = \
            self.do_object_detection(image, is_return_image=False)

        return (final_boxes, final_probs, final_class)

    def convert_data_to_image(self, data, width, height, depth=4):
        """Take the 1D array of image data, convert to 3D image array of w x h x d.
        The 1D image data is a sequence of uint8 type B,G,R values for line 1, then line 2, ..."""
        #print('data shape : {}'.format(data.shape))
        if depth == 4:
            data = data.reshape((height, width, 4))
            ret_data = data[:,:,:3]
        else:  # We assume depth is 3 in this case
            ret_data = data.reshape((height, width, 3))
            print(ret_data.shape)
        return ret_data

    def convert_data_to_image_matlab(self, data, width, height):
        """Take the 1D array of image data, convert to 3D image array of w x h x d.
        The 1D image data is a sequence of uint8 type B,G,R values for line 1, then line 2, ..."""
        #print('data shape : {}'.format(data.shape))
        ret_data = self.convert_data_to_image(np.array(data), width, height, depth=3)
        return ret_data.tolist()

    def do_object_detection_on_file(self, filename):
        """Read image from the given file, do the object detection.
        Return detection image."""
        image = cv2.imread(filename)
        image = image.astype(np.float32, copy=False)
        (im_out, det_list) = self.do_object_detection(image)
        return (image, det_list)

    def do_object_detection(self, image, is_return_image=True):
        """Do the object detection using SqueezeDet."""
        # Resize image to the image size in the trained model.
        start_time = timeit.default_timer()
        image = cv2.resize(image,
                           (self.model_config.IMAGE_WIDTH, self.model_config.IMAGE_HEIGHT))
        elapsed_time = timeit.default_timer() - start_time
        # print('Classifier image resize: {}'.format(elapsed_time))
        # Normalize image by subtracting the mean B,G,R values of the training set.
        input_image = image - self.model_config.BGR_MEANS
        # Detect
        det_boxes, det_probs, det_class = self.sess.run(
            [self.model.det_boxes, self.model.det_probs, self.model.det_class],
            feed_dict={self.model.image_input:[input_image]})
        # Filter
        final_boxes, final_probs, final_class = self.model.filter_prediction(
            det_boxes[0], det_probs[0], det_class[0])
        # Remove the detected objects which has less than threshold probability.
        keep_idx = [idx for idx in range(len(final_probs)) \
                    if final_probs[idx] > self.model_config.PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]

        if is_return_image:
            # Draw boxes
            _draw_box(image,
                      final_boxes,
                      [self.model_config.CLASS_NAMES[idx]+': (%.2f)'% prob
                       for idx, prob in zip(final_class, final_probs)],
                      cdict=self.class_color_dict)
        else:
            image = None
        return image, (final_boxes, final_probs, final_class)

    def do_object_detection_for_matlab(self, image):
        """Do the object detection using SqueezeDet."""
        # Resize image to the image size in the trained model.
        # start_time = timeit.default_timer()
        # image = cv2.resize(image,
        #                    (self.model_config.IMAGE_WIDTH, self.model_config.IMAGE_HEIGHT))
        # elapsed_time = timeit.default_timer() - start_time
        # print('Classifier image resize: {}'.format(elapsed_time))
        # Normalize image by subtracting the mean B,G,R values of the training set.
        input_image = image - self.model_config.BGR_MEANS
        # Detect
        det_boxes, det_probs, det_class = self.sess.run(
            [self.model.det_boxes, self.model.det_probs, self.model.det_class],
            feed_dict={self.model.image_input:[input_image]})
        # Filter
        final_boxes, final_probs, final_class = self.model.filter_prediction(
            det_boxes[0], det_probs[0], det_class[0])
        # Remove the detected objects which has less than threshold probability.
        keep_idx = [idx for idx in range(len(final_probs)) \
                    if final_probs[idx] > self.model_config.PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]

        return (final_boxes, final_probs, final_class)

    def box_to_relative_position(self, object_class, detection_box, camera_width_px):
        """Convert detection box to longitudinal and lateral distance to the object."""
        height_px = max(detection_box[self.DET_BOX_HEIGHT_IND], 1)
        if object_class == self.PEDESTRIAN_CLASS_LABEL:
            long_dist = self.PED_DISTANCE_SCALE_FACTOR / height_px
            m_per_pixel = self.KNOWN_PED_HEIGHT / height_px
        else:
            long_dist = self.CAR_DISTANCE_SCALE_FACTOR / height_px
            m_per_pixel = self.KNOWN_CAR_HEIGHT / height_px
        lat_dist = m_per_pixel * (detection_box[self.DET_BOX_X_IND] - (camera_width_px / 2))
        return [-lat_dist, long_dist]

    def save_output_to_file(self, output_file_name, image):
        """Save the image to the file."""
        if not tf.gfile.Exists(FILE_PATH + '/data/out/'):
            tf.gfile.MakeDirs(FILE_PATH + '/data/out/')
        file_name = os.path.split(output_file_name)[1]
        out_file_name = os.path.join(FILE_PATH + '/data/out/', 'out_'+file_name)
        cv2.imwrite(out_file_name, image)
        # print('Image detection output saved to {}'.format(out_file_name))

    def close_session(self):
        """Close the tensorFlow session."""
        self.sess.close()


def main():
    """Run classifier standalone to classify images from an input directory."""
    classifier = Classifier()
    classifier.start_classification_engine()
    for file_name in glob.iglob(FILE_PATH + '/data/webots.png'):
        im_out = classifier.do_object_detection_on_file(file_name)
        classifier.save_output_to_file(file_name, im_out)
    classifier.close_session()


if __name__ == "__main__":
    main()
