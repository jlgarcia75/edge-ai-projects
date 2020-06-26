#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 07:54:34 2020

@author: jesusgarcia
Adapted from: https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
"""


import numpy as np
import os
#import six.moves.urllib as urllib
import sys
#import tarfile
import tensorflow as tf
#import zipfile
import pathlib
from datetime import datetime



if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')



 
#from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
#from PIL import Image
 
import cv2
cap = cv2.VideoCapture("/Users/jesusgarcia/Dropbox/Education/Udacity/Project 1 - People Counter/Pedestrian_Detect_2_1_1.mp4")
 

#os.chdir('models/research/')
#print("We are in dir", os.getcwd())
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def find_accuracy(results):
    truth = np.loadtxt('/Users/jesusgarcia/Dropbox/Education/Udacity/Project 1 - People Counter/truth.csv', delimiter=",")
    acc = np.equal(np.array(results),truth).sum()/len(truth)*100
    return acc


MODEL_NAME = '/Users/jesusgarcia/Dropbox/Education/Udacity/Project 1 - People Counter/ssd_mobilenet_v2_coco_2018_03_29'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/Users/jesusgarcia/Dropbox/AI/tf/models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
 
NUM_CLASSES = 90
predictions = []
infer_duration_ms = 0

""" 
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
"""

start_time = datetime.now()
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
 
load_duration_ms = (datetime.now() - start_time).total_seconds() * 1000

#label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

frame_count = 0
with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph) as sess:
    while cap.isOpened():
        flag, image_bgr = cap.read()
        if not flag:
            break
        frame_count+=1
        image_np = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
        start_time = datetime.now()
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        infer_duration_ms += (datetime.now() - start_time).total_seconds()
        predictions.append(int(num_detections))
    # Visualization of the results of a detection.
    """
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        cv2.imshow('object detection', cv2.resize(image_rgb, (800,600)))
        if cv2.waitKey(25) and 0xFF == ord('q'):
            break
    """
cap.release()
cv2.destroyAllWindows()

outF = open("tf_stats.txt", "a")
now = datetime.now()
outF.write("TensorFlow Results")
outF.write("\nModel : {}".format("ssd_mobilenet_v2_coco_2018_03_29"))
outF.write ("\nCurrent date and time : \n")
outF.write(now.strftime("%Y-%m-%d %H:%M:%S"))
outF.write("\nAccuracy: {}%".format(find_accuracy(predictions)))
outF.write("\nLoad Model Time: {:.2f} ms".format(load_duration_ms))
outF.write("\nTotal Inference Time: {:.2f} ms".format(infer_duration_ms*1000))
outF.write("\nAverage Inference Time: {:.2f} ms".format(infer_duration_ms/frame_count*1000))
outF.write("\n\n\n*********************************************************************************\n\n\n")
outF.close()
