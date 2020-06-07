# Computer Pointer Controller

This is the third project for the Udacity course Intel® Edge AI for IoT Developers. The purpose of this project is to use multiple deep learning models to move a mouse cursor on a screen using eye and head pose from a webcam or video.

## Project Set Up and Installation
### Prerequisites
* Intel® OpenVINO™ Toolkit 2019.R3 or above
* pandas
* numpy
* cv2 (comes with OpenVINO)

### Setup
1.  Clone git into your working directory.
2.  Source the OpenVINO environment.
3.  


## Demo
Two scripts are provided that will run the project with a sample demo video or with your webcam.
### Linux or MacOS
  * rundemo.sh
  * runcam.sh
### Windows
  * rundemo.bat
  * runcam.bat

## Documentation
The program has many command line arguments that allow you to customize how it runs.

usage: main.py [-h] -i INPUT [-p PRECISIONS] [-fdm FD_MODEL] [-flm FL_MODEL]
               [-hpm HP_MODEL] [-gem GE_MODEL] [-l CPU_EXTENSION] [-d DEVICE]
               [-pt PROB_THRESHOLD] [-bm BENCHMARK] [-nf NUM_FRAMES]
               [-sv SHOWVIDEO] [-async ASYNCINFER]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to input image or video file. 0 for webcam.
  -p PRECISIONS, --precisions PRECISIONS
                        Set model precisions as a comma-separated list without
                        spaces, e.g. FP32,FP16,FP32-INT8 (FP16 by default)
  -fdm FD_MODEL, --fd_model FD_MODEL
                        Path to directory for a trained Face Detection
                        model.The directory must have the model precisions as
                        subdirectories.(../models/intel/face-detection-adas-
                        binary-0001 by default)
  -flm FL_MODEL, --fl_model FL_MODEL
                        Path to directory for a trained Facial Landmarks
                        model.The directory must have the model precisions as
                        subdirectories.../models/intel/landmarks-regression-
                        retail-0009 by default)
  -hpm HP_MODEL, --hp_model HP_MODEL
                        Path to directory for a trained Head Pose model.The
                        directory must have the model precisions as
                        subdirectories.(../models/intel/head-pose-estimation-
                        adas-0001 by default)
  -gem GE_MODEL, --ge_model GE_MODEL
                        Path to directory for a trained Gaze Detection
                        model.The directory must have the model precisions as
                        subdirectories.(../models/intel/gaze-estimation-
                        adas-0002 by default)
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with thekernels impl.
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable. The program will look for a
                        suitable plugin for the device specified (CPU by
                        default)
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for detections filtering (0.3 by
                        default)
  -bm BENCHMARK, --benchmark BENCHMARK
                        Show benchmark data? True|False (True by default)
  -nf NUM_FRAMES, --num_frames NUM_FRAMES
                        The number of frames to run. Use this to limit running
                        time, especially if using webcam. (100 by default)
  -sv SHOWVIDEO, --showvideo SHOWVIDEO
                        Show video while running? True|False. (True by
                        default)
  -async ASYNCINFER, --asyncinfer ASYNCINFER
                        Run asynchronouse inference landmarks and head pose
                        models? True|False. (True by default)

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
