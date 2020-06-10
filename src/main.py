"""Move Mouse Pointer."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import cv2
import pandas as pd
import numpy as np
from sys import exit
from datetime import datetime
import time
#from face_detection import FaceDetection
#from facial_landmarks import FacialLandmarks
#from head_pose import HeadPose
from model_base import ModelBase
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController
from MediaReader import MediaReader
from signal import SIGINT, signal
from argparse import ArgumentParser
from sys import platform
#from matplotlib import pyplot as pl
import os


# Get correct CPU extension
if platform == "linux" or platform == "linux2":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
elif platform == "darwin":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
elif platform == "win32":
    CPU_EXTENSION = None
else:
    print("Unsupported OS.")
    exit(1)

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to input image or video file. 0 for webcam.")
    parser.add_argument("-p", "--precisions", required=False, type=str, default='FP16',
                                        help="Set model precisions as a comma-separated list without spaces"
                                           ", e.g. FP32,FP16,FP32-INT8 (FP16 by default)")
    parser.add_argument("-fdm", "--fd_model", required=False, type=str,
                        help="Path to directory for a trained Face Detection model."
                        "This directory path must include the model's precision because"
                        "face-detection-adas-binary-0001 has only one precision, FP32-INT1."
                        "(../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001"
                        " by default)",
                        default="../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001")
    parser.add_argument("-flm", "--fl_model", required=False, type=str,
                        help="Path to directory for a trained Facial Landmarks model."
                        "The directory must have the model precisions as subdirectories."
                        "../models/intel/landmarks-regression-retail-0009 by default)",
                        default="../models/intel/landmarks-regression-retail-0009")
    parser.add_argument("-hpm", "--hp_model", required=False, type=str,
                        help="Path to directory for a trained Head Pose model."
                        "The directory must have the model precisions as subdirectories."
                        "(../models/intel/head-pose-estimation-adas-0001 by default)",
                        default="../models/intel/head-pose-estimation-adas-0001")
    parser.add_argument("-gem", "--ge_model", required=False, type=str,
                        help="Path to directory for a trained Gaze Detection model."
                        "The directory must have the model precisions as subdirectories."
                        "(../models/intel/gaze-estimation-adas-0002 by default)",
                        default="../models/intel/gaze-estimation-adas-0002")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, required=False, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. The program "
                             "will look for a suitable plugin for the device "
                             "specified (CPU by default)")
    parser.add_argument("-ct", "--conf_threshold", type=float, default=0.3, required=False,
                        help="Probability threshold for detections filtering"
                        " (0.3 by default)")
    parser.add_argument("-bm", "--benchmark", required=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    default=True, help="Show benchmark data? True|False (True by default)")
    parser.add_argument("-nf", "--num_frames", required=False, type=int, default=100,
                    help="The number of frames to run. Use this to limit running time, "
                    "especially if using webcam. (100 by default)")
    parser.add_argument("-sv", "--showvideo", required=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    default=True, help="Show video while running? True|False. (True by default)")
    parser.add_argument("-async", "--async_inference", required=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    default=True, help="If True, run asynchronous inference where possible."
                                        "If false, run synchronous inference. True|False. (True by default)")

    return parser


def draw_box(image, start_point, end_point):
    box_col = (0,255,0) #GREEN
    thickness = 4
    image = cv2.rectangle(image, start_point, end_point, box_col, thickness)
    return image

def scale_dims(shape, x, y):
    width = shape[1]
    height= shape[0]
    x = int(x*width)
    y = int(y*height)

    return x, y

#scale the landmarks to the whole frame size
def scale_landmarks(landmarks, image_shape, orig):
    color = (0,255,0) #GREEN
    thickness = cv2.FILLED
    num_lm = len(landmarks)
    orig_x = orig[0]
    orig_y = orig[1]
    scaled_landmarks = []
    for point in range(0, num_lm, 2):
        x, y = scale_dims(image_shape, landmarks[point], landmarks[point+1])
        x_scaled = orig_x + x
        y_scaled = orig_y + y
    #    image = cv2.circle(image, (x_scaled, y_scaled), 2, color, thickness)
        scaled_landmarks.append([x_scaled, y_scaled])
    #return scaled_landmarks, image
    return scaled_landmarks

def process_model_names(name):
        new_path = name.replace("\\","/")
        dir, new_name = new_path.rsplit('/', 1)
        if name.find(dir) == -1:
            dir, _ = name.rsplit('\\',1)
        return dir, new_name

def run_pipeline(network, input_image, duration):
    # Detect faces
    #Preprocess the input

    start_time = time.perf_counter()
    p_image = network.preprocess_input(input_image)
    duration['input'] += time.perf_counter() - start_time
    #print("duration ", duration['input']*100000)

    #Infer the faces
    start_time = time.perf_counter()
    network.sync_infer(p_image)
    duration['infer'] += time.perf_counter() - start_time

    #Get the outputs
    start_time = time.perf_counter()
    output = network.preprocess_output()
    duration['output'] += time.perf_counter() - start_time

    return duration, output

def output_bm(args, t_df, r_df, frames):
    avg_df = t_df/frames
    now = datetime.now()
    print("OpenVINO Results")
    print ("Current date and time: ",now.strftime("%Y-%m-%d %H:%M:%S"))
    print("Platform: {}".format(platform))
    print("Device: {}".format(args.device))
    print("Asynchronous Inference: {}".format(args.async_inference))
    print("Precision: {}".format(args.precisions))
    print("Total frames: {}".format(frames))
    print("Total runtimes(s):")
    print(r_df)
    print("\nTotal Durations(ms) per phase:")
    print(t_df)
    print("\nDuration(ms)/Frames per phase:")
    print(avg_df)
    print("\n*********************************************************************************\n\n\n")
def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    """
    try:
        ######### Setup fonts for text on image ########################

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (10,40)
        fontScale = .5
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 1 px
        thickness = 1
        text = ""
        #######################################

        fd_dir, fd_model = process_model_names(args.fd_model)
        _, fl_model = process_model_names(args.fl_model)
        _, hp_model = process_model_names(args.hp_model)
        _, ge_model = process_model_names(args.ge_model)


         # Initialize the classes
        fd_infer_network = ModelBase(name='face detection', dev=args.device, ext=args.cpu_extension, threshold=args.conf_threshold)
        fl_infer_network = ModelBase(name = 'landmark detection', dev=args.device, ext=args.cpu_extension)
        hp_infer_network = ModelBase(name = 'head pose', dev=args.device, ext=args.cpu_extension)
        ge_infer_network = GazeEstimation(name = 'gaze estimation',dev=args.device, ext=args.cpu_extension)

        precisions=args.precisions.split(",")

        if args.benchmark:
            columns=['load','input','infer','output']
            model_indeces=[fd_infer_network.short_name, fl_infer_network.short_name, hp_infer_network.short_name, ge_infer_network.short_name]
            iterables = [model_indeces,precisions]
            index = pd.MultiIndex.from_product(iterables, names=['Model','Precision'])
            total_df = pd.DataFrame(np.zeros((len(model_indeces)*len(precisions),len(columns)), dtype=float),index=index, columns=columns)

        flip=False

        cap = MediaReader(args.input)
        if cap.sourcetype() == MediaReader.CAMSOURCE:
            flip = True

        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        mc = MouseController('high', 'fast')
        screenWidth, screenHeight = mc.monitor()
        if args.showvideo:
            cv2.startWindowThread()
            cv2.namedWindow("Out")
            cv2.moveWindow("Out", int((screenWidth-frame_width)/2), int((screenHeight+frame_height)/2))
        mc.put(int(screenWidth/2), int(screenHeight/2)) #Place the mouse cursor in the center of the screen
        # Process frames until the video ends, or process is exited
        ### TODO: Load the models through `infer_network` ###
        print("Video being shown: ", str(args.showvideo))
        #Dictionary to store runtimes for each precision
        runtime={}
        for precision in precisions:
            print("Beginning test for precision {}.".format(precision))
            frame_count=0
            runtime_start = time.perf_counter()
            #fd_dir = os.path.join(args.fd_model, precision)
            fl_dir = os.path.join(args.fl_model, precision)
            hp_dir = os.path.join(args.hp_model, precision)
            ge_dir = os.path.join(args.ge_model, precision)
            total_df.loc(axis=0)[fd_infer_network.short_name,precision]['load'] = fd_infer_network.load_model(dir=fd_dir, name=fd_model)
            total_df.loc(axis=0)[fl_infer_network.short_name,precision]['load'] = fl_infer_network.load_model(dir=fl_dir, name=fl_model)
            total_df.loc(axis=0)[hp_infer_network.short_name,precision]['load'] = hp_infer_network.load_model(dir=hp_dir, name=hp_model)
            total_df.loc(axis=0)[ge_infer_network.short_name,precision]['load'] = ge_infer_network.load_model(dir=ge_dir, name=ge_model)

            too_many = False
            not_enough = False
            single = False
            gaze = [[0, 0, 0]]
            cap.set(property=cv2.CAP_PROP_POS_FRAMES, val=0)
            while cap.isOpened():

                if args.num_frames!=None and frame_count>=args.num_frames:
                    break

                # Read the next frame
                flag, frame = cap.read()
                if not flag:
                    break

                #Flip the frame is the input is from the web cam
                if flip: frame=cv2.flip(frame, 1)

                frame_count+=1

                frame = cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
                if args.showvideo: cv2.imshow("Out", frame)

                # Break if escape key pressed
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                # Detect faces
                total_df.loc(axis=0)[fd_infer_network.short_name,precision], outputs =  run_pipeline(fd_infer_network, frame, total_df.loc(axis=0)[fd_infer_network.short_name,precision])

                coords = [[x_min, y_min, x_max, y_max] for _, _, conf, x_min, y_min, x_max, y_max in outputs[fd_infer_network.output_name][0][0] if conf>=args.conf_threshold]
                num_detections = len(coords)

                ### Execute the pipeline only if one face is in the frame
                if num_detections == 1:
                    too_many = False
                    not_enough = False
                    if not single:
                        text="I see you. Move the mouse cursor with your eyes."
                        single=True

                    x_min, y_min, x_max, y_max = coords[0]
                    orig_x, orig_y = scale_dims(frame.shape, x_min, y_min)
                    x_max, y_max = scale_dims(frame.shape, x_max, y_max)
                    #frame = draw_box(frame,(orig_x, orig_y), (x_max, y_max))
                    cropped_frame = frame[orig_y:y_max, orig_x:x_max]

                    if args.async_inference: #Run asynchronous inference
                    #facial landmark detection preprocess the input
                        start_time = time.perf_counter()
                        frame_for_input = fl_infer_network.preprocess_input(cropped_frame)
                        total_df.loc(axis=0)[fl_infer_network.short_name,precision]['input']  += time.perf_counter() - start_time

                        #Run landmarks inference asynchronously
                        # do not measure time, not relevant since it is asynchronous
                        fl_infer_network.predict(frame_for_input)

                        #Send cropped frame to head pose estimation
                        start_time = time.perf_counter()
                        frame_for_input = hp_infer_network.preprocess_input(cropped_frame)
                        total_df.loc(axis=0)[hp_infer_network.short_name,precision]['input']  += time.perf_counter() - start_time

                        #Head pose infer
                        hp_infer_network.predict(frame_for_input)

                        #Wait for async inferences to complete
                        if fl_infer_network.wait()==0:
                            start_time = time.perf_counter()
                            outputs = fl_infer_network.preprocess_output()
                            scaled_lm = scale_landmarks(landmarks=outputs[fl_infer_network.output_name][0], image_shape=cropped_frame.shape, orig=(orig_x, orig_y))
                            total_df.loc(axis=0)[fl_infer_network.short_name,precision]['output']  += time.perf_counter() - start_time

                        if hp_infer_network.wait()==0:
                            start_time = time.perf_counter()
                            outputs = hp_infer_network.preprocess_output()
                            hp_angles = [outputs['angle_y_fc'][0], outputs['angle_p_fc'][0], outputs['angle_r_fc'][0]]
                            total_df.loc(axis=0)[hp_infer_network.short_name,precision]['output'] += time.perf_counter() - start_time
                    else: #Run synchronous inference
                        #facial landmark detection preprocess the input
                        total_df.loc(axis=0)[fl_infer_network.short_name,precision], outputs = run_pipeline(fl_infer_network, cropped_frame, total_df.loc(axis=0)[fl_infer_network.short_name,precision])
                        scaled_lm = scale_landmarks(landmarks=outputs[fl_infer_network.output_name][0], image_shape=cropped_frame.shape, orig=(orig_x, orig_y))
                        #Send cropped frame to head pose estimation
                        total_df.loc(axis=0)[hp_infer_network.short_name, precision], outputs = run_pipeline(hp_infer_network, cropped_frame, total_df.loc(axis=0)[hp_infer_network.short_name,precision])
                        hp_angles = [outputs['angle_y_fc'][0], outputs['angle_p_fc'][0], outputs['angle_r_fc'][0]]

                    input_duration, predict_duration, output_duration, gaze = ge_infer_network.sync_infer(face_image=frame, landmarks=scaled_lm, head_pose_angles=[hp_angles])
                    total_df.loc(axis=0)[ge_infer_network.short_name,precision]['input'] += input_duration
                    total_df.loc(axis=0)[ge_infer_network.short_name,precision]['infer'] += predict_duration
                    total_df.loc(axis=0)[ge_infer_network.short_name,precision]['output'] += output_duration
                    #Move the mouse cursor
                    mc.move(gaze[0][0], gaze[0][1])

                elif num_detections > 1:
                    single = False
                    not_enough=False
                    if not too_many:
                        text="Too many faces confuse me. I need to see only one face."
                        too_many=True
                else:
                    too_many = False
                    single=False
                    if not not_enough:
                        text="Is there anybody out there?"
                        not_enough=True

            ## End While Loop
            runtime[precision] = time.perf_counter() - runtime_start
            # Release the capture and destroy any OpenCV windows
            print("Completed run for precision {}.".format(precision))
            if args.benchmark:
                rt_df = pd.DataFrame.from_dict(runtime, orient='index', columns=["Total runtime"])
                rt_df['FPS'] = frame_count/rt_df["Total runtime"]

        ### End For Loop
        cap.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        #Collect Stats
        #Setup dataframe
        if args.benchmark:
            output_bm(args, total_df, rt_df, frame_count)

    except KeyboardInterrupt:
        #Collect Stats
        print("Detected keyboard interrupt")
        if args.benchmark:
            if args.benchmark:
                rt_df = pd.DataFrame.from_dict(runtime, orient='index', columns=["Total runtime"])
                rt_df['FPS'] = frame_count/rt_df["Total runtime"]
                output_bm(args, total_df, rt_df, frame_count)
        leave_program()
    except Exception as e:
         print("Exception: ",e)
         leave_program()

def leave_program():
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    exit()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
#    signal(SIGINT, sig_handler)
    # Grab command line args
    args = build_argparser().parse_args()

    # Perform inference on the input stream
    infer_on_stream(args)

if __name__ == '__main__':
    main()
