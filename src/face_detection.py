'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import time
import cv2
import sys
import os

class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, dev, ext=None):
        self.model = None
        self.core = None
        self.device = dev
        self.extensions= ext


    def load_model(self, dir, name):
        '''
        TODO: This method needs to be completed by you
        Returns: Time to load model
        '''
        self.model_structure=os.path.join(dir, name+".xml")
        self.model_weights=os.path.join(dir, name+".bin")

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you entered the correct model path? {}".format(e))

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

        start_time = time.time()
        self.core = IECore()
        #Check for unsupported layers
        self.check_model()
        self.exec_net = self.core.load_network(self.model, self.device, num_requests=0)
        duration_ms = time.time() - start_time
        # Add an extension, if applicable
        if self.extensions:
            self.core.add_extension(self.extensions, self.device)



        return duration_ms

    def predict(self, image, conf_threshold=0.3):
        '''
        TODO: This method needs to be completed by you
        Returns: Duration of inference time, new image with detections
        '''
        #Run Inference
        self.exec_net.infer({self.input_name:image})
        return



    def check_model(self):
        '''
        Check for unsupported layers

        Returns
        -------
        None.

        '''
        core = IECore()
         # Get the supported layers of the network
        supported_layers = core.query_network(network=self.model, device_name=self.device)

        # Check for any unsupported layers, and let the user
        # know if anything is missing. Exit the program, if so.

        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("The following layers are not supported by the plugin for specified device {}:\n {}".format(self.device, ', '.join(unsupported_layers)))
            print(f"Please try to specify {self.device} extensions library path in sample's command line parameters using -l or --extension command line argument.")
            sys.exit(1)


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        n, c, h, w = self.input_shape

        new_image = cv2.resize(image, (w, h))
        new_image = new_image.transpose((2,0,1))
        new_image = new_image.reshape(n, c, h, w)

        return new_image

    def preprocess_output(self, threshold=0.3):
        '''
        TODO: This method needs to be completed by you
        Creates array of box coordinates from outputs

        '''
        #Get the outputs
        outputs = self.exec_net.requests[0].outputs[self.output_name]
        coords = []
        for box in outputs[0][0]:
            _, _, conf, x_min, y_min, x_max, y_max = box
            if conf >= float(threshold):
                coords.append([x_min, y_min, x_max, y_max])

        return coords
