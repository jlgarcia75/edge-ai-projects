"""People Counter."""
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
import re
import sys
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
    parser.add_argument("-fdm", "--fd_model", required=True, type=str,
                        help="Path to an xml file with a trained Face Detection model.")
    parser.add_argument("-flm", "--fl_model", required=True, type=str,
                        help="Path to an xml file with a trained Facial Landmark model.")
    parser.add_argument("-hpm", "--hp_model", required=True, type=str,
                        help="Path to an xml file with a trained Head Pose model.")
    parser.add_argument("-gem", "--ge_model", required=True, type=str,
                        help="Path to an xml file with a trained Gaze Detection model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=CPU_EXTENSION,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.3,
                        help="Probability threshold for detections filtering"
                        "(0.3 by default)")
    parser.add_argument("-bm", "--benchmark", required=False, type=bool, default=False,
                    help="See benchmark data True|False.")
    parser.add_argument("-fancy", "--fancy", required=False, type=bool, default=False,
                    help="See benchmark data in a fancy report.")
    parser.add_argument("-p", "--precisions", required=False, type=str, default='FP16',
                    help="Set model precisions.")

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

        precisions=args.precisions.split(",")

        fd_infer_duration_ms  = 0
        fd_input_duration_ms  = 0
        fd_output_duration_ms  = 0

        fl_infer_duration_ms  = 0
        fl_input_duration_ms = 0
        fl_output_duration_ms = 0

        hp_infer_duration_ms  = 0
        hp_input_duration_ms = 0
        hp_output_duration_ms = 0

        ge_infer_duration_ms  = 0
        ge_input_duration_ms = 0
        ge_output_duration_ms = 0

        frame_count=0


        fd_dir, fd_model = process_model_names(args.fd_model)
        fl_dir, fl_model = process_model_names(args.fl_model)
        hp_dir, hp_model = process_model_names(args.hp_model)
        ge_dir, ge_model = process_model_names(args.ge_model)


         # Initialise the classes
        fd_infer_network = FaceDetection(dev=args.device, ext=args.cpu_extension)
        fl_infer_network = FacialLandmarks(dev=args.device, ext=args.cpu_extension)
        hp_infer_network = HeadPose(dev=args.device, ext=args.cpu_extension)
        ge_infer_network = GazeEstimation(dev=args.device, ext=args.cpu_extension)


        ### TODO: Load the models through `infer_network` ###
        fd_load_duration_ms = fd_infer_network.load_model(dir=fd_dir, name=fd_model)
        fl_load_duration_ms = fl_infer_network.load_model(dir=fl_dir, name=fl_model)
        hp_load_duration_ms = hp_infer_network.load_model(dir=hp_dir, name=hp_model)
        ge_load_duration_ms = ge_infer_network.load_model(dir=ge_dir, name=ge_model)

        #Get gaze input description

        flip=False

        cap = MediaReader(args.input)
        if cap.sourcetype() == MediaReader.CAMSOURCE:
            flip = True

        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        mc = MouseController('high', 'fast')
        screenWidth, screenHeight = mc.monitor()
        cv2.startWindowThread()
        cv2.namedWindow("Out")
        cv2.moveWindow("Out", int((screenWidth-frame_width)/2), int((screenHeight+frame_height)/2))
        mc.put(int(screenWidth/2), int(screenHeight/2)) #Place the mouse cursor in the center of the screen
        # Process frames until the video ends, or process is exited
        too_many = False
        not_enough = False
        single = False
        gaze = [[0, 0, 0]]
        while cap.isOpened():
            # Read the next frame
            flag, frame = cap.read()
            if not flag:
                break

            #Flip the frame is the input is from the web cam
            if flip: frame=cv2.flip(frame, 1)

            frame_count+=1
            cv2.imshow("Out", frame)

            # Break if escape key pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
                    print("I see you. Move the mouse cursor with your eyes.")
                    single=True

                x_min, y_min, x_max, y_max = coords[0]
                orig_x, orig_y = scale_dims(frame.shape, x_min, y_min)
                x_max, y_max = scale_dims(frame.shape, x_max, y_max)
                #frame = draw_box(frame,(orig_x, orig_y), (x_max, y_max))
                cropped_frame = frame[orig_y:y_max, orig_x:x_max]

                #Preprocess the input
                start_time = time()
                frame_for_input = fl_infer_network.preprocess_input(cropped_frame)
                fl_input_duration_ms = time() - start_time

                #Run landmarks inference asynchronously
                fl_infer_start_time = time()
                fl_infer_network.predict(frame_for_input)

                #Send cropped frame to head pose estimation
                start_time = time()
                frame_for_input = hp_infer_network.preprocess_input(cropped_frame)
                hp_input_duration_ms += start_time

                hp_infer_start_time = time()
                hp_infer_network.predict(frame_for_input)

                #Wait for async inferences to complete
                if fl_infer_network.wait()==0:
                    fl_infer_duration_ms += time() - fl_infer_start_time
                    start_time = time()
                    landmarks = fl_infer_network.preprocess_output()
                    scaled_lm = scale_landmarks(landmarks=landmarks[0], image_shape=cropped_frame.shape, orig=(orig_x, orig_y))
                    fl_output_duration_ms = time() - start_time

                if hp_infer_network.wait()==0:
                    hp_infer_duration_ms += time() - hp_infer_start_time
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
                    print("Too many faces confuse me. I need to see only one face.")
                    too_many=True
            else:
                too_many = False
                single=False
                if not not_enough:
                    print("Is there anybody out there?")
                    not_enough=True

        ## End While Loop
        # Release the capture and destroy any OpenCV windows
        print("Video is over. Good bye!")
        cap.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        #Collect Stats
        #Setup dataframe
        if args.benchmark:
            model_indeces=['facial detection', 'landmark detection', 'head pose', 'gaze estimation']
            metric_columns={'load model':[fd_load_duration_ms*1000, fl_load_duration_ms*1000, hp_load_duration_ms*1000, ge_load_duration_ms*1000],
                            'input prep':[fd_input_duration_ms*1000, fl_input_duration_ms*1000, hp_input_duration_ms*1000, ge_input_duration_ms*1000],
                            'total predict':[fd_infer_duration_ms*1000, fl_infer_duration_ms*1000, hp_infer_duration_ms*1000, ge_infer_duration_ms*1000],
                            'average predict':[fd_infer_duration_ms*1000/frame_count, fl_infer_duration_ms*1000/frame_count, hp_infer_duration_ms*1000/frame_count, ge_infer_duration_ms*1000/frame_count],
                            'fetch output':[fd_output_duration_ms*1000, fl_output_duration_ms*1000, hp_output_duration_ms*1000, ge_output_duration_ms*1000],
                            }
            df = pd.DataFrame(metric_columns, index=model_indeces)
            outF = open("gaze_stats.txt", "a")
            now = datetime.now()
            outF.write("OpenVINO Results\n")
            outF.write ("\nCurrent date and time:\n")
            outF.write(now.strftime("%Y-%m-%d %H:%M:%S"))
            outF.write("\nPlatform: {}".format(platform))
            outF.write("\nDevice: {}".format(args.device))
            outF.write("\nProbably Threshold: {}".format(args.prob_threshold))
            outF.write("Precision: {}".format(args.precisions))
            outF.write("Total frames: {}".format(frame_count))
            print(df)
            outF.write("\n\n*********************************************************************************\n\n\n")
            outF.close()
        '''
        outF.write("\n\nFace Detection Stats")
        outF.write("\nModel: {}".format(args.fd_model))
        outF.write("\nModel Load Time: {:.2f} ms".format(fd_load_duration_ms*1000))
        outF.write("\nTotal Inference Time: {:.2f} ms".format(fd_infer_duration_ms*1000))
        outF.write("\nAverage Inference Time: {:.2f} ms".format(fd_infer_duration_ms*1000/frame_count))

        outF.write("\n\nLandmark Detection Stats")
        outF.write("\nModel: {}".format(args.fl_model))
        outF.write("\nModel Load Time: {:.2f} ms".format(fl_load_duration_ms*1000))
        outF.write("\nTotal Inference Time: {:.2f} ms".format(fl_infer_duration_ms*1000))
        outF.write("\nAverage Inference Time: {:.2f} ms".format(fl_infer_duration_ms*1000/frame_count))

        outF.write("\n\nHead Pose Estimation Stats")
        outF.write("\nModel: {}".format(args.hp_model))
        outF.write("\nModel Load Time: {:.2f} ms".format(hp_load_duration_ms*1000))
        outF.write("\nTotal Inference Time: {:.2f} ms".format(hp_infer_duration_ms*1000))
        outF.write("\nAverage Inference Time: {:.2f} ms".format(hp_infer_duration_ms*1000/frame_count))

        outF.write("\n\nGaze Estimation Stats")
        outF.write("\nModel: {}".format(args.ge_model))
        outF.write("\nModel Load Time: {:.2f} ms".format(ge_load_duration_ms*1000))
        outF.write("\nTotal Inference Time: {:.2f} ms".format(ge_infer_duration_ms*1000))
        outF.write("\nAverage Inference Time: {:.2f} ms".format(ge_infer_duration_ms*1000/frame_count))
        '''
    except KeyboardInterrupt:
        #Collect Stats
        print("Detected keyboard interrupt")
        if args.benchmark:
            model_indeces=['facial detection', 'landmark detection', 'head pose', 'gaze estimation']
            metric_columns={'load model':[fd_load_duration_ms*1000, fl_load_duration_ms*1000, hp_load_duration_ms*1000, ge_load_duration_ms*1000],
                            'input prep':[fd_input_duration_ms*1000, fl_input_duration_ms*1000, hp_input_duration_ms*1000, ge_input_duration_ms*1000],
                            'predict':[fd_infer_duration_ms*1000, fl_infer_duration_ms*1000, hp_infer_duration_ms*1000, ge_infer_duration_ms*1000],
                            'fetch output':[fd_output_duration_ms*1000, fl_output_duration_ms*1000, hp_output_duration_ms*1000, ge_output_duration_ms*1000],
                            }

            total_df = pd.DataFrame(metric_columns, index=model_indeces)
            avg_df = total_df/frame_count
            print("df ", total_df)
            print("avg df", avg_df)
            outF = open("gaze_stats.txt", "a")
            if args.fancy:
                from pandas_profiling import ProfileReport
                profile = ProfileReport(total_df, "Gaze Estimation Profile Report", explorative=True)
                profile.to_file('gaze_profile_report.html')

            now = datetime.now()
            outF.write("OpenVINO Results\n")
            outF.write ("\nCurrent date and time:\n")
            outF.write(now.strftime("%Y-%m-%d %H:%M:%S"))
            outF.write("\nPlatform: {}".format(platform))
            outF.write("\nDevice: {}".format(args.device))
            outF.write("\nProbably Threshold: {}".format(args.prob_threshold))
            outF.write("Precision: {}".format(args.precisions))
            outF.write("Total frames: {}".format(frame_count))
            outF.write("\n\n*********************************************************************************\n\n\n")
            outF.close()
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
