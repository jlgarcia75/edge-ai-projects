'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import time
import cv2
import sys
import os
from model_base import ModelBase

class GazeEstimation(ModelBase):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, name, dev, ext=None):
        ModelBase.__init__(self, name, dev, ext)
        # self.model = None
        # self.core = None
        # self.device=dev
        # self.extensions= ext
        self.left_eye_input = 'left_eye_image'
        self.right_eye_input = 'right_eye_image'
        self.head_pose_input = 'head_pose_angles'

    def sync_infer(self, face_image, landmarks, head_pose_angles):
        '''
        TODO: This method needs to be completed by you
        Returns: Duration of input processing time, inference time, gaze vector
        '''
        #Preprocess the input
        start_time = time.perf_counter()
        self.input_shape = self.model.inputs[self.left_eye_input].shape
        eye_offset = int(self.input_shape[2]/2)
        right_eye = landmarks[0]
        left_eye = landmarks[1]
        left_crop = face_image[left_eye[1]-eye_offset:left_eye[1]+eye_offset, left_eye[0]-eye_offset:left_eye[0]+eye_offset]
        right_crop = face_image[right_eye[1]-eye_offset:right_eye[1]+eye_offset, right_eye[0]-eye_offset:right_eye[0]+eye_offset]
        eye_images = list(map(self.preprocess_input,[left_crop, right_crop]))
        input_duration_ms = time.perf_counter() - start_time

        #Run Inference
        start_time = time.perf_counter()
        self.exec_net.infer({self.left_eye_input:eye_images[0], self.right_eye_input:eye_images[1], self.head_pose_input:head_pose_angles})
        infer_duration_ms = time.perf_counter() - start_time

        #Get the outputs
        start_time = time.perf_counter()
        detections = self.exec_net.requests[0].outputs[self.output_name]
        output_duration_ms = time.perf_counter() - start_time

        return input_duration_ms, infer_duration_ms, output_duration_ms, detections
