# Zepei Luo
# 20321548
# scyzl9@nottingham.edu.cn

'''
This python file define the classes and methods to finish the panorama generation task.
All works are done by the author, but some idea from the following sources are referenced (not using code directly) and learned:

Stitching two pictures: https://pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
Smoothing the picture after stitching: https://github.com/tranleanh/image-panorama-stitching/blob/master/panorama.py
'''

import os
import math
import numpy as np
import cv2 as cv
from abc import ABC, abstractmethod
from utils import extract_frames


# This abstract class contains the essential functions to finish panorama generation
# Due to the fact that stitch towards left and right are different and 
# "towards left" is much harder (the figure to be stitched that obtained after warpPerspective surpasses the boundary),
# there will have to sub-classes to inherit and implement separately
class Stitcher(ABC):
    def __init__(self):
        # Defines the distance of "transition region", used in where the mask is created to smooth the picture
        # source_video1: 25
        # source_video2: 80
        # source_video3: 100
        # source_video5: 30  (interval=40)
        self.MASK_OFFSET = 60
        # Defines the parameter that used to determine high-quality feature matches, called Lowe's in CV area
        # Its value in the range [0.7, 0.8] will have best matching result.
        self.__ratio = 0.75
        # Defines the maximum pixel "wiggle room" allowed by the RANSAC algorithm
        self.__reprojThresh = 4.0

    #  visualizes matching keypoints between two frames and return the comparison diagram
    def display_matches(self, frame1, frame2, key_points_1, key_points_2, ideal_match, S):
        (h1, w1) = frame1.shape[:2]
        (h2, w2) = frame2.shape[:2]
        pic_matches = np.zeros((max(h1, h2), w1 + w2, 3), dtype='uint8')
        pic_matches[0:h1, 0:w1] = frame1
        pic_matches[0:h2, w1:] = frame2
        for ((trainIdx, queryIdx), n) in zip(ideal_match, S):
            if n == 1:
                point1 = (int(key_points_1[queryIdx][0]), int(key_points_1[queryIdx][1]))
                point2 = (int(key_points_2[trainIdx][0]) + w1, int(key_points_2[trainIdx][1]))
                cv.line(pic_matches, point1, point2, (0, 255, 0), 1)

        return pic_matches
    
    # This method would finds keypoints and extracts local invariant descriptors of the given frame
    # Difference of Gaussian (DoG) keypoint detector and the SIFT feature extractor are applied.
    def detect_describe(self, frame):
        # Convert one color space to another color space
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        sift = cv.SIFT_create()
        (key_points, descriptors) = sift.detectAndCompute(gray, None)
        
        key_points = np.float32([kp.pt for kp in key_points])
        return key_points, descriptors
    
    # This method is used to match the features together
    # It would computes the distances between each pair of descriptors, 
    # loops over the descriptors from both pictures, 
    # and finds the lowest distance between them.
    def match_key_points(self, key_points_1, key_points_2, descriptors_1, descriptors_2):
        # Initialize BruteForce, using the default parameters
        # raw knn matches calculation
        descriptor_matcher = cv.DescriptorMatcher_create('BruteForce')
        knn_matches = descriptor_matcher.knnMatch(descriptors_1, descriptors_2, 2) 

        # Get ideal match
        ideal_match = []
        for m in knn_matches:
            if len(m) == 2 and m[0].distance < self.__ratio * m[1].distance:
                ideal_match.append((m[0].trainIdx, m[0].queryIdx))

        # A minimum of four matches are needed to compute a homography.
        if len(ideal_match) > 4:
            points1 = np.float32([key_points_1[i] for (_, i) in ideal_match])
            points2 = np.float32([key_points_2[i] for (i, _) in ideal_match])

            # find the homography between the point1 and point2, S means the status of each matched point
            (homography_matrix, S) = cv.findHomography(points1, points2, cv.RANSAC, self.__reprojThresh)
            return (ideal_match, homography_matrix, S)
        # Return None if at least four are not met
        return None

    # The main body of the stitch method
    @abstractmethod
    def stitch(self, frames):
        pass
    
    # Generate the mask that could smooth the picture after stitching
    # It can generate masks for both the left frame and the right frame
    @abstractmethod
    def mask_generation(self, frame1, frame2, type_mask):
        pass
    
    # This method would remove the black (useless) part of the picture obtained by stitching
    @abstractmethod
    def remove_black_borders(self, frame):
        pass


# This class will stitch frame1 to the right side of frame2
class Stitcher_Right(Stitcher):
    
    def __init__(self):
        super().__init__()
    
    def stitch(self, frames):
        (frame2, frame1) = frames
        
        # Gets key points and descriptors
        (key_points_1, descriptors_1) = super().detect_describe(frame1)
        (key_points_2, descriptors_2) = super().detect_describe(frame2)
        
        Mask = super().match_key_points(key_points_1, key_points_2, descriptors_1, descriptors_2)
        if Mask is None:
            print("Mask is none!!! Matching Failed.")
            return frame2
        
        # Otherwise, use a perspective warp to "sew" the pictures together.
        (ideal_match, homography_matrix, S) = Mask
        
        stitch_outcome_height = frame1.shape[0]
        stitch_outcome_width = frame1.shape[1] + frame2.shape[1]
        
        # Generate the left part of the stitched picture
        mask_for_left_part = self.mask_generation(frame2, frame1, type_mask = "Left")
        stitch_outcome_left_part = np.zeros((stitch_outcome_height, stitch_outcome_width, 3))
        stitch_outcome_left_part[0:frame2.shape[0], 0:frame2.shape[1], :] = frame2
        stitch_outcome_left_part_masked = stitch_outcome_left_part * mask_for_left_part
        
        # Generate the right part of the stitched picture
        stitch_outcome_right_part = cv.warpPerspective(frame1, homography_matrix, (stitch_outcome_width, stitch_outcome_height))
        mask_for_right_part = self.mask_generation(frame2, frame1, type_mask = "Right")
        stitch_outcome_right_part_masked = stitch_outcome_right_part * mask_for_right_part
        
        stitch_outcome = stitch_outcome_left_part_masked + stitch_outcome_right_part_masked
        stitch_outcome=stitch_outcome.astype(np.uint8)
        
        final_outcome = self.remove_black_borders(stitch_outcome)
        
        #pic_matches = super().display_matches(frame1, frame2, key_points_1, key_points_2, ideal_match, S)
        #return final_outcome, pic_matches
        return final_outcome
        
    # Generate the mask that could smooth the picture after stitching
    # The biggest difference between the "left version" is highlighted
    # The transition is achieved using a linear gradient.
    def mask_generation(self, frame1, frame2, type_mask):
        stitch_outcome_height = frame1.shape[0]
        # Main difference 1!!!
        stitch_outcome_width = frame1.shape[1] + frame2.shape[1]
        # Main difference 2!!!
        mask_barrier = frame1.shape[1] - self.MASK_OFFSET
        
        blend_mask = np.zeros((stitch_outcome_height, stitch_outcome_width))
    
        if type_mask == "Left":
            linear_gradient_right = np.tile(np.linspace(1, 0, 2 * self.MASK_OFFSET).T, (stitch_outcome_height, 1))
            blend_mask[:, mask_barrier - self.MASK_OFFSET:mask_barrier + self.MASK_OFFSET] = linear_gradient_right
            blend_mask[:, :mask_barrier - self.MASK_OFFSET] = 1
        elif type_mask == "Right":
            linear_gradient_left = np.tile(np.linspace(0, 1, 2 * self.MASK_OFFSET).T, (stitch_outcome_height, 1))
            blend_mask[:, mask_barrier - self.MASK_OFFSET: mask_barrier + self.MASK_OFFSET] = linear_gradient_left
            blend_mask[:, mask_barrier + self.MASK_OFFSET:] = 1
            
        return cv.merge([blend_mask, blend_mask, blend_mask])    
    
    
    # Step1: find the contour of the main body
    # Step2: traverse the first row of pixels from right to left 
    #        until the color is not pure black, deleting the right side
    def remove_black_borders(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # the grayscale image is converted to a binary image using threshold operations
        # Distinguish between non-black areas of the image and black backgrounds
        _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(largest_contour)
            
            for n in reversed(range(w)):
                if frame[0, n, 0] == 0 and frame[0, n, 1] == 0 and frame[0, n, 2] == 0:
                    continue
                else:
                    sub_num = n         
                    break
            
            # print(sub_num)
            cropped_frame = frame[y:y+h, x:x+sub_num]
            return cropped_frame
        # Return the original if no contours found
        return frame


# This class will stitch frame2 to the left side of frame1
class Stitcher_Left(Stitcher):
    
    def __init__(self):
        super().__init__()
        
    # expand the area of the right frame, which will be stitched
    # this is the biggest difficult compared with "Stitcher_Right"
    # this function can overcome the question:
    # frame2 (left) will surpass the boundary after its perspective is warped
    def expand_right_frame(self, frame1, frame2):
        # expand frame2 in this function
        right_frame_height, _ , num_channels = frame2.shape
        # Determine the left fill width to add
        expanding_width = frame1.shape[1]
        expanding_area = np.zeros((right_frame_height, expanding_width, num_channels), dtype=np.uint8)
        expanded_right_frame = np.concatenate((expanding_area, frame2), axis=1)
        return expanded_right_frame

    def stitch(self, frames):
        (frame2, frame1) = frames

        expanded_frame1 = self.expand_right_frame(frame2, frame1)
        
        # Gets key points and descriptors
        (key_points_1, descriptors_1) = super().detect_describe(expanded_frame1)
        (key_points_2, descriptors_2) = super().detect_describe(frame2)
        
        # mention the order is differnt with the right version
        Mask = super().match_key_points(key_points_2, key_points_1, descriptors_2, descriptors_1)
        if Mask is None:
            print("Mask is none!!! Matching Failed.")
            return frame1
        
        # Otherwise, use a perspective warp to "sew" the pictures together.
        (ideal_match, homography_matrix, S) = Mask
        
        stitch_outcome_height = frame2.shape[0]
        stitch_outcome_width = expanded_frame1.shape[1]  # different with "right" since frame1 has been expanded
        
        stitch_outcome_left_part = cv.warpPerspective(frame2, homography_matrix, (stitch_outcome_width, stitch_outcome_height))
        mask_for_left_part = self.mask_generation(expanded_frame1, frame2, type_mask = "Left")
        stitch_outcome_left_part_masked = stitch_outcome_left_part * mask_for_left_part
        
        mask_for_right_part = self.mask_generation(expanded_frame1, frame2, type_mask = "Right")
        stitch_outcome_right_part = np.zeros((stitch_outcome_height, stitch_outcome_width, 3))
        stitch_outcome_right_part[0:expanded_frame1.shape[0], 0:expanded_frame1.shape[1], :] = expanded_frame1
        stitch_outcome_right_part_masked = stitch_outcome_right_part * mask_for_right_part
        
        stitch_outcome = stitch_outcome_left_part_masked + stitch_outcome_right_part_masked
        stitch_outcome=stitch_outcome.astype(np.uint8)
        
        final_outcome = self.remove_black_borders(stitch_outcome)
        
        #return final_outcome, pic_matches
        return final_outcome
        
    # Generate the mask that could smooth the picture after stitching
    # The biggest difference between the "right version" is highlighted
    # The transition is achieved using a linear gradient.
    def mask_generation(self, frame1, frame2, type_mask):
        stitch_outcome_height = frame1.shape[0]
        # Main difference 1!!! Since frame1 has been expanded
        stitch_outcome_width = frame1.shape[1]
        # Main difference 2!!!
        mask_barrier = self.MASK_OFFSET + frame2.shape[1]
        
        blend_mask = np.zeros((stitch_outcome_height, stitch_outcome_width))
    
        if type_mask == "Left":
            linear_gradient_right = np.tile(np.linspace(1, 0, 2 * self.MASK_OFFSET).T, (stitch_outcome_height, 1))
            blend_mask[:, mask_barrier - self.MASK_OFFSET: mask_barrier + self.MASK_OFFSET] = linear_gradient_right
            blend_mask[:, : mask_barrier - self.MASK_OFFSET] = 1
        elif type_mask == "Right":
            linear_gradient_left = np.tile(np.linspace(0, 1, 2 * self.MASK_OFFSET).T, (stitch_outcome_height, 1))
            blend_mask[:, mask_barrier - self.MASK_OFFSET: mask_barrier + self.MASK_OFFSET] = linear_gradient_left
            blend_mask[:, mask_barrier + self.MASK_OFFSET:] = 1
            
        return cv.merge([blend_mask, blend_mask, blend_mask])   
        
    # Step1: find the contour of the main body
    # Step2: traverse the first row of pixels from right to left 
    #        until the color is not pure black, deleting the right side
    # Step3: remove the black part generated by joggle (small degree shaking only)
    def remove_black_borders(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # the grayscale image is converted to a binary image using threshold operations
        # Distinguish between non-black areas of the image and black backgrounds
        _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        
        # Step1
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            # x, y are the coordinates of the top left point of the matrix
            # w, h are the weight and height of the matrix
            x, y, w, h = cv.boundingRect(largest_contour)
            
            # Step2
            for n in range(w):
                if frame[0, n, 0] == 0 and frame[0, n, 1] == 0 and frame[0, n, 2] == 0:
                    continue
                else:
                    begin_num = n
                    break
                
            #print(begin_num)
            cropped_frame_1 = frame[y:y+h, begin_num: x+w]
                
            # Step3
            for m in reversed(range(h)):
                if cropped_frame_1[m, 0, 0] == 0 and cropped_frame_1[m, 0, 1] == 0 and cropped_frame_1[m, 0, 2] == 0:
                    continue
                else:
                    num = m
                    break
            
            cropped_frame_2 = cropped_frame_1[:num, :]
            return cropped_frame_2
        # Return the original if no contours found
        return frame
        
    
def alignment_height(fig1, fig2):
    fig1_height = fig1.shape[0]
    fig2_height = fig2.shape[0]
    
    if fig1_height == fig2_height:
        return fig1, fig2
    elif fig1_height > fig2_height:
        diff = fig1_height - fig2_height
        top = math.ceil(diff/2)
        sub = diff - top
        aligned_fig1 = fig1[top: fig1_height-sub, :]
        return aligned_fig1, fig2
    elif fig1_height < fig2_height:
        diff = fig2_height - fig1_height
        top = math.ceil(diff/2)
        sub = diff - top
        aligned_fig2 = fig2[top: fig2_height-sub, :]
        return fig1, aligned_fig2
    

def panorama_gen(video_path, progress_callback=None):
    interval = 40
    frames_stack = extract_frames(video_path, interval)
    length = len(frames_stack)
    # print(length)

    # VERSION 1 (the far right side has severe distortion)
    # final_figure = frames_stack[0]
    # for n in range(1, length):
    #     stitched = Stitcher()
    #     final_figure = stitched.stitch([final_figure, frames_stack[n]])
    
    mid_frame_pos = math.ceil(length / 2.0)
    print(mid_frame_pos)
    left_frames_stack = []
    for i in range(mid_frame_pos):
        left_frames_stack.append(frames_stack.pop())
    
    while not (len(frames_stack) == 0 or len(left_frames_stack) == 0):
        left_frame_fix = left_frames_stack.pop()
        right_frame_stitch = frames_stack.pop()
        left_frame_fix_aligned, right_frame_stitch_aligned = alignment_height(left_frame_fix, right_frame_stitch)
        
        stitcher_right = Stitcher_Right()
        result_right = stitcher_right.stitch([left_frame_fix_aligned, right_frame_stitch_aligned])
        frames_stack.append(result_right)
        
        if len(frames_stack) == 0 or len(left_frames_stack) == 0:
            break
        left_frame_stitch = left_frames_stack.pop()
        right_frame_fix = frames_stack.pop()
        left_frame_stitch_aligned, right_frame_fix_aligned = alignment_height(left_frame_stitch, right_frame_fix)
        
        stitcher_left = Stitcher_Left()
        result_left = stitcher_left.stitch([left_frame_stitch_aligned, right_frame_fix_aligned])
        left_frames_stack.append(result_left)
        print(len(frames_stack))
        print(len(left_frames_stack))
        progress_callback((mid_frame_pos - len(frames_stack)) / mid_frame_pos)

    if len(frames_stack) == 0:
        panorama = left_frames_stack.pop()
    elif len(left_frames_stack) == 0:
        panorama = frames_stack.pop()
        
    panorama = panorama[:, :panorama.shape[1] - 1200]

    directory_path = os.path.dirname(video_path)
    panorama_path = os.path.join(directory_path, "panorama.png")
    panorama_path = panorama_path.replace("\\", "/")
    print(panorama_path)
    cv.imwrite(panorama_path, panorama)
    progress_callback(1)
    return panorama_path
    
# def panorama_gen_test(video_path):
#     img1 = cv.imread('./source_video/p0.png')
#     img2 = cv.imread('./source_video/p50.png')

#     stitcher_right = Stitcher_Right()
#     #stitcher_right = Stitcher_Left()
#     final_outcome = stitcher_right.stitch([img1, img2])

#     cv.imwrite('./source_video/new/AAA.png', final_outcome)
#     #cv.imwrite('./source_video/result_vis.png', pic_matches)
    
# Test
#panorama_gen("C:/Users/Magi/Desktop/CV_code/source_video5/source_video5.mp4")
#panorama_gen_test("C:/Users/Magi/Desktop/CV_code/source_video/source_video1.mp4")
