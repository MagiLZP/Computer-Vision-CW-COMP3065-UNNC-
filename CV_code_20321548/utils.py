# Zepei Luo
# 20321548
# scyzl9@nottingham.edu.cn

# pip install imutils

import cv2 as cv
import imutils

# In this work, False will be printed, which means the author's env is not cv3
def check_opencv_version():
    print(imutils.is_cv3())
    

# Extracts frames from a video file at specified intervals.
# The task is then accomplished by implementing the concatenation of these frames
def extract_frames(video_path, interval):
    video = cv.VideoCapture(video_path)
    success, frame = video.read()
    
    if success == False:
        print("Unsuccess to read the video, please check the path")
    
    n = 0
    frames = []
    while success:
        if n % interval == 0:
            frames.append(frame)
            # Check the frames obtained
            cv.imwrite('./source_video2/p{}.png'.format(n), frame)
        success, frame = video.read()
        n += 1
    video.release()
    
    return frames[::-1]