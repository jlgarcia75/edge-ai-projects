#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:41:54 2020

@author: jesusgarcia
"""

import cv2
from argparse import ArgumentParser

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to input video.")
    parser.add_argument("-o","--outputdir", required=True, type=str,
                        help="Output directory.")
    return parser

def main():
    print("ARE WE IN")
    # Grab command line args
    # args = build_argparser().parse_args()
    cap = cv2.VideoCapture("Pedestrian_Detect_2_1_1.mp4")
    cap.open("Pedestrian_Detect_2_1_1.mp4")
    num=0
    while cap.isOpened():
        # Read the next frame"
        num+=1
        flag, frame = cap.read()
        if not flag:
            break
    
        cv2.imshow("Movie", frame)
        cv2.imwrite("outputframes/frame_" + str(num),frame)
        
        key_pressed = cv2.waitKey(60)
        if key_pressed == 27:
            break
    
    
    cap.release()
    cv2.destroyAllWindows()
