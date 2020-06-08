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
from time import time
from face_detection import FaceDetection
from facial_landmarks import FacialLandmarks
from head_pose import HeadPose
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
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.3, required=False,
                        help="Probability threshold for detections filtering"
                        " (0.3 by default)")
    parser.add_argument("-bm", "--benchmark", required=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    default=True, help="Show benchmark data? True|False (True by default)")
    parser.add_argument("-nf", "--num_frames", required=False, type=int, default=100,
                    help="The number of frames to run. Use this to limit running time, "
                    "especially if using webcam. (100 by default)")
    parser.add_argument("-sv", "--showvideo", required=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    default=True, help="Show video while running? True|False. (True by default)")

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
        precisions=args.precisions.split(",")

        if args.benchmark:
            columns=['load model','input prep','predict','fetch output']
            model_indeces=['facial detection', 'landmark detection', 'head pose', 'gaze estimation']
            iterables = [model_indeces,precisions]
            index = pd.MultiIndex.from_product(iterables, names=['Model','Precision'])
            total_df = pd.DataFrame(index=index, columns=columns)

        fd_infer_duration_ms  = 0
        fd_input_duration_ms  = 0
        fd_output_duration_ms  = 0

        fl_input_duration_ms = 0
        fl_output_duration_ms = 0

        hp_input_duration_ms = 0
        hp_output_duration_ms = 0

        ge_infer_duration_ms  = 0
        ge_input_duration_ms = 0
        ge_output_duration_ms = 0



        fd_dir, fd_model = process_model_names(args.fd_model)
        _, fl_model = process_model_names(args.fl_model)
        _, hp_model = process_model_names(args.hp_model)
        _, ge_model = process_model_names(args.ge_model)


         # Initialise the classes
        fd_infer_network = FaceDetection(dev=args.device, ext=args.cpu_extension)
        fl_infer_network = FacialLandmarks(dev=args.device, ext=args.cpu_extension)
        hp_infer_network = HeadPose(dev=args.device, ext=args.cpu_extension)
        ge_infer_network = GazeEstimation(dev=args.device, ext=args.cpu_extension)

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
            runtime_start = time()
            #fd_dir = os.path.join(args.fd_model, precision)
            fl_dir = os.path.join(args.fl_model, precision)
            hp_dir = os.path.join(args.hp_model, precision)
            ge_dir = os.path.join(args.ge_model, precision)
            fd_load_duration_ms = fd_infer_network.load_model(dir=fd_dir, name=fd_model)
            fl_load_duration_ms = fl_infer_network.load_model(dir=fl_dir, name=fl_model)
            hp_load_duration_ms = hp_infer_network.load_model(dir=hp_dir, name=hp_model)
            ge_load_duration_ms = ge_infer_network.load_model(dir=ge_dir, name=ge_model)

            too_many = False
            not_enough = False
            single = False
            gaze = [[0, 0, 0]]
            cap.set(property=cv2.CAP_PROP_POS_FRAMES, val=0)
            while cap.isOpened():

                if args.num_frames!=None and frame_count>args.num_frames:
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
                #Preprocess the input
                start_time = time()
                p_frame = fd_infer_network.preprocess_input(frame)
                fd_input_duration_ms = time() - start_time

                #Infer the faces
                start_time = time()
                fd_infer_network.predict(p_frame)
                fd_infer_duration_ms += time() - start_time

                #Get the outputs
                start_time = time()
                coords = fd_infer_network.preprocess_output(threshold = args.prob_threshold)
                fd_output_duration_ms += time() - start_time

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

                    #facial landmar detection preprocess the input
                    start_time = time()
                    frame_for_input = fl_infer_network.preprocess_input(cropped_frame)
                    fl_input_duration_ms = time() - start_time

                    #Run landmarks inference asynchronously
                    # do not measure time, not relevant since it is asynchronous
                    fl_infer_network.predict(frame_for_input)

                    #Send cropped frame to head pose estimation
                    start_time = time()
                    frame_for_input = hp_infer_network.preprocess_input(cropped_frame)
                    hp_input_duration_ms += time() - start_time

                    #Head pose infer
                    hp_infer_network.predict(frame_for_input)

                    #Wait for async inferences to complete
                    if fl_infer_network.wait()==0:
                        start_time = time()
                        landmarks = fl_infer_network.preprocess_output()
                        scaled_lm = scale_landmarks(landmarks=landmarks[0], image_shape=cropped_frame.shape, orig=(orig_x, orig_y))
                        fl_output_duration_ms = time() - start_time

                    if hp_infer_network.wait()==0:
                        start_time = time()
                        y, p, r = hp_infer_network.preprocess_output()
                        hp_output_duration_ms = time() - start_time

                        input_duration, predict_duration, output_duration, gaze = ge_infer_network.predict(face_image=frame, landmarks=scaled_lm, head_pose_angles=[[y, p, r]])
                        ge_input_duration_ms += input_duration
                        ge_infer_duration_ms += predict_duration
                        ge_output_duration_ms += output_duration
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
            runtime[precision] = time() - runtime_start
            # Release the capture and destroy any OpenCV windows
            print("Completed run for precision {}.".format(precision))
            if args.benchmark:
                rt_df = pd.DataFrame.from_dict(runtime, orient='index', columns=["Total runtime"])
                rt_df["Average runtime/frame"] = rt_df["Total runtime"]/frame_count
                metric_columns=np.array([[fd_load_duration_ms*1000, fl_load_duration_ms*1000, hp_load_duration_ms*1000, ge_load_duration_ms*1000],
                            [fd_input_duration_ms*1000, fl_input_duration_ms*1000, hp_input_duration_ms*1000, ge_input_duration_ms*1000],
                            [fd_infer_duration_ms*1000, None, None, ge_infer_duration_ms*1000],
                            [fd_output_duration_ms*1000, fl_output_duration_ms*1000, hp_output_duration_ms*1000, ge_output_duration_ms*1000]
                            ]).T
                total_df.loc(axis=0)[:,precision] = metric_columns

        ### End For Loop
        cap.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        #Collect Stats
        #Setup dataframe
        if args.benchmark:
            avg_df = total_df/frame_count
            now = datetime.now()
            print("OpenVINO Results")
            print ("Current date and time: ",now.strftime("%Y-%m-%d %H:%M:%S"))
            print("Platform: {}".format(platform))
            print("Device: {}".format(args.device))
            print("Probably Threshold: {}".format(args.prob_threshold))
            print("Precision: {}".format(args.precisions))
            print("Total frames: {}".format(frame_count))
            print("Total runtimes:")
            print(rt_df)
            print("\nTotal Durations per phase(ms):")
            print(total_df)
            print("\nDuration (ms) per phase /Frame:")
            print(avg_df)
            print("\n*********************************************************************************\n\n\n")
    except KeyboardInterrupt:
        #Collect Stats
        print("Detected keyboard interrupt")
        if args.benchmark:
            rt_df = pd.DataFrame.from_dict(runtime, orient='index', columns=["Total runtime"])
            rt_df["Average runtime/frame"] = rt_df["Total runtime"]/frame_count
            metric_columns=np.array([[fd_load_duration_ms*1000, fl_load_duration_ms*1000, hp_load_duration_ms*1000, ge_load_duration_ms*1000],
                        [fd_input_duration_ms*1000, fl_input_duration_ms*1000, hp_input_duration_ms*1000, ge_input_duration_ms*1000],
                        [fd_infer_duration_ms*1000, None, None, ge_infer_duration_ms*1000],
                        [fd_output_duration_ms*1000, fl_output_duration_ms*1000, hp_output_duration_ms*1000, ge_output_duration_ms*1000]
                        ]).T
            total_df.loc(axis=0)[:,precision] = metric_columns
            avg_df = total_df/frame_count

            now = datetime.now()
            print("OpenVINO Results\n")
            print ("Current date and time: ",now.strftime("%Y-%m-%d %H:%M:%S"))
            print("Platform: {}".format(platform))
            print("Device: {}".format(args.device))
            print("Probably Threshold: {}".format(args.prob_threshold))
            print("Precision: {}".format(args.precisions))
            print("Total frames: {}".format(frame_count))
            print("Total runtimes:")
            print(rt_df)
            print("\nTotal Durations per phase(ms):")
            print(total_df)
            print("\nDuration(ms) per phase/Frame:")
            print(avg_df)
            print("\n*********************************************************************************\n\n\n")
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
