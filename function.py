#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:46:29 2019

@author: linux
"""

# Functions
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
 # for list of images

#%matplotlib inline
#!clear

#read Image
def readIMG(img_name):
    image = mpimg.imread(img_name)
    shape = image.shape
    color_img = np.copy(image)    
    
    return color_img, shape

# n Image Plot  -> lt.imshow(image, 'gray_r') INVERT THE GRAY REPRESENTATION
def plot_n(img, title, cmap=''):
    plt.figure(figsize=(15,12))
    #plt.figure(figsize=(9,7))
    plt.subplots_adjust(wspace=0.14, hspace=0.25)
        
    if len(img) % 2 > 0:
        raw = (len(img)-1) / 2 + 1
        print('raw =' + str(raw))
    else:
        raw = len(img) / 2
        print('raw =' + str(raw))
        
    for i in range(len(img)):

        plt.subplot(raw, 2 , i+1)
        plt.imshow(img[i],cmap=str(cmap))
        plt.title(title[i], fontsize = 15)
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
# =============================================================================
# #print image
# def ims(image):
#     plt.imshow(image, 'gray')
#     plt.show()
#     print(image.shape)
# =============================================================================
    
# Define color selection criteria
def threshold(rgb_threshold, color_img, plot_flag):
    
    # Do a boolean or with the "|" character to identify
    # Mask pixels below the threshold
    color_thres_img = (color_img[:,:,0] < rgb_threshold[0]) | \
                        (color_img[:,:,1] < rgb_threshold[1]) | \
                        (color_img[:,:,2] < rgb_threshold[2]) 
                    
    if(plot_flag):
        plt.imshow(color_thres_img, cmap='gray_r')
        plt.show()
        
    
    
    return color_thres_img
     
# =============================================================================
#                       0|                |
#                        |                |
#                        |                |
#                        |    y_line      |
#                        x_left --- x_right
#                        |     /   \      |
#                        |    /     \     |
#                        |   /       \    |
#                       0|   ---------    |960
#                     540

def polygon_roi(img, imshape, bottom_left, bottom_right, y_line, x_left, x_right, rgb_threshold, plot_flag):
    
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(img)   
    ignore_mask_color = 255

    vertices = np.array([[(bottom_left,imshape[0]),(x_left, y_line), (x_right, y_line), \
                          (bottom_right,imshape[0])]], dtype=np.int32) 
    
    #vertices = np.array([[(0,imshape[0]),(x_left, y_line), (x_right, y_line), \
    #                      (imshape[1],imshape[0])]], dtype=np.int32)
    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(img, mask)
# =============================================================================
#     # Mask pixels below the threshold
#     color_thresholds_polygon = (masked_edges[:,:,0] < rgb_threshold[0]) | \
#                                 (masked_edges[:,:,1] < rgb_threshold[1]) | \
#                                 (masked_edges[:,:,2] < rgb_threshold[2])
# =============================================================================
                        
    print('Vertices.shape'+str(vertices.shape))
    
    if(plot_flag):
        roi_marked_img = cv2.polylines(img, vertices, True , (255,0,0),3 )
        
        #plot Images
        images = [roi_marked_img, masked_edges]
        titels = ['ROI marked Image','masked_edges']
        plot_n(images,titels,'gray_r')

        # save the ROI Image
        #img = cv2.cvtColor(roi_marked_img,cv2.COLOR_RGB2BGR)
        #write_name = 'output_images/ROI.png'
        #cv2.imwrite(write_name,img)

    return masked_edges, vertices


def mask_img(vertices, shape, color_img, bin_image):
    color_select   = np.copy(color_img)
    line_image     = np.copy(color_img)
    
    fit_left = np.polyfit(( vertices[0,0,0],  vertices[0,1,0]), (vertices[0,0,1], vertices[0,1,1]), 1)
    fit_right = np.polyfit((vertices[0,3,0],  vertices[0,2,0]), (vertices[0,3,1], vertices[0,2,1]), 1)
    fit_bottom = np.polyfit((vertices[0,0,0], vertices[0,3,0]), (vertices[0,0,1], vertices[0,3,1]), 1)

    # Find the region inside the lines
    XX, YY = np.meshgrid(np.arange(0, shape[1]), np.arange(0, shape[0]))
    region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                        (YY > (XX*fit_right[0] + fit_right[1])) & \
                        (YY < (XX*fit_bottom[0] + fit_bottom[1]))

    # Mask color selection
    color_select[bin_image | region_thresholds] = [0,0,0]
    
    # Find where image is both colored right and in the region
    line_image[~bin_image & region_thresholds] = [255,0,0]
    
    return color_select, line_image



def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#def cany_edge(gray_img, l_thresh, h_thresh, k_size):
#    
#    # Cany Parameter: l_thresh -> low threshold; h_thresh -> high threshold; 
#    # Gaussian smoothing parameter: k_size -> kernel size
#    # Define a kernel size for Gaussian smoothing / blurring
#    blur_gray = cv2.GaussianBlur(gray_img,(k_size, k_size), 0)
#
#    
#    return cv2.Canny(blur_gray, l_thresh, h_thresh)

def cany_edge(gray_img, l_thresh, h_thresh):
    #cv2.Canny() applies a 5x5 Gaussian internally
    
    return cv2.Canny(gray_img, l_thresh, h_thresh)

def hough_transform(cany_img, rho, theta, threshold, min_line_length, max_line_gap):
    
    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(cany_img, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    
    return lines


print('---------------Function finish----------------')
























