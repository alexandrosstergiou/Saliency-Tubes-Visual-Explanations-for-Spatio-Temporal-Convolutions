import cv2
import numpy as np
import matplotlib.pyplot as plt
import os,glob
import copy
from moviepy.editor import *


def frames_extractor(file,frame_num,step):
    frames = []
    try:
        clip = VideoFileClip(file)
    except:
        if clip in locals():
            clip.close()
        print('FFMPEG COULD NOT INPORT: '+file)
        return
    count = 1
    i = 0
    for frame in clip.iter_frames():

        if (i >= frame_num):
            break
        y,x,c = frame.shape
        
        img = frame[y/2-112:y/2+112,x/2-112:x/2+112]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.true_divide(img,255.0)

        if (count % step == 0):
            frames.append(copy.deepcopy(img))
            i+=1

        count+=1

    clip.close()
    del clip.reader
    del clip

    if (len(frames) == frame_num):
        return frames
    else:
        return
