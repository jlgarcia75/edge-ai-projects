'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import time
import cv2
import logging as log
import sys

class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device, threshold, extensions=None):


        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.extensions= extensions
        self.network = None
        self.core = None


        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you entered the correct model path? {}".format(e))

        self.left_eye_input = 'left_eye_image'
        self.right_eye_input = 'right_eye_image'
        self.head_pose_input = 'head_pose_angles'
        self.eye_shape=self.model.inputs[self.left_eye_input].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape


    def get_eye_input_shape(self):
        return self.eye_shape

    def load_model(self):
        '''
        TODO: This method needs to be completed by you
        Returns: Time to load model
        '''
        start_time = time.time()
        self.core = IECore()

        #Check for unsupported layers
        self.check_model()

        self.exec_net = self.core.load_network(self.model, self.device, num_requests=0)

        duration_ms = time.time() - start_time
        # Add an extension, if applicable
        if self.extensions:
            self.plugin.add_extension(self.extensions, self.device)

        return duration_ms

    def predict(self, face_image, landmarks, head_pose_angles):
        '''
        TODO: This method needs to be completed by you
        Returns: Duration of input processing time, inference time, gaze vector
        '''
        #Preprocess the input
        start_time = time.time()
        eye_offset = int(self.eye_shape[2]/2)
        right_eye = landmarks[0]
        left_eye = landmarks[1]
        left_crop = face_image[left_eye[1]-eye_offset:left_eye[1]+eye_offset, left_eye[0]-eye_offset:left_eye[0]+eye_offset]
        right_crop = face_image[right_eye[1]-eye_offset:right_eye[1]+eye_offset, right_eye[0]-eye_offset:right_eye[0]+eye_offset]
        left_image = self.preprocess_input(image=left_crop, shape=self.eye_shape)
        right_image = self.preprocess_input(image=right_crop, shape=self.eye_shape)
        input_duration_ms = time.time() - start_time

        #Run Inference
        start_time = time.time()
        self.exec_net.infer({self.left_eye_input:left_image, self.right_eye_input:right_image, self.head_pose_input:head_pose_angles})
        infer_duration_ms = time.time() - start_time

        #Get the outputs
        start_time = time.time()
        detections = self.exec_net.requests[0].outputs[self.output_name]
        output_duration_ms = time.time() - start_time

        return input_duration_ms, infer_duration_ms, output_duration_ms, detections

    def check_model(self):
        '''
        Check for unsupported layers

        Returns
        -------
        None.

        '''
         # Get the supported layers of the network
        supported_layers = self.core.query_network(network=self.model, device_name=self.device)

        # Check for any unsupported layers, and let the user
        # know if anything is missing. Exit the program, if so.

        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("The following layers are not supported by the plugin for specified device {}:\n {}".format(self.device, ', '.join(unsupported_layers)))
            print(f"Please try to specify {self.device} extensions library path in sample's command line parameters using -l or --extension command line argument")
            sys.exit(1)


    def preprocess_input(self, image, shape):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        n, c, h, w = shape

        new_image = cv2.resize(image, (w, h))
        new_image = new_image.transpose((2,0,1))
        new_image = new_image.reshape(n, c, h, w)

        return new_image
