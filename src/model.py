'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class Model_X:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        raise NotImplementedError

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        raise NotImplementedError

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def check_model(model, device):
        core = IECore()
         # Get the supported layers of the network
        supported_layers = core.query_network(network=model, device_name=device)

        # Check for any unsupported layers, and let the user
        # know if anything is missing. Exit the program, if so.

        unsupported_layers = [l for l in model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("The following layers are not supported by the plugin for specified device {}:\n {}".format(device, ', '.join(unsupported_layers)))
            print(f"Please try to specify {device} extensions library path in sample's command line parameters using -l or --extension command line argument.")
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

    def preprocess_output(self, outputs):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        raise NotImplementedError
