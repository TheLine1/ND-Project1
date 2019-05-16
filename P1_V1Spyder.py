#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projekt1 Udacity Nanodegree Self-Driving-Car
"""
#%%
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os   # for list of images
import function as f
import plotIMG as p
import matplotlib.gridspec as gridspec

#%%
#!clear
#read Image
img, shape = f.readIMG('test_images/solidWhiteRight.jpg')

img_color_orig = np.copy(img)
line_image     = np.copy(img)
cany_image     = np.copy(img)


# Define color selection criteria Mache daraus ein binary 0 & 1
rgb_threshold, color_thres_img = f.threshold(200,200,200,img_color_orig)
                    
# ROI - Four Side Polygone -> polygon_roi(img, imshape, y_line, x_left, x_right, rgb_threshold)      
y_line  = 310
x_left  = 430
x_right = 530

masked_edges, color_thresholds_polygon, vertices \
            = f.polygon_roi(img, shape, y_line, x_left, x_right, rgb_threshold)

color_select, line_image = f.mask_img(vertices, shape, img_color_orig, color_thres_img)

# Display Image with ROI and color selection Lanes

x = [vertices[0,0,0], vertices[0,1,0], vertices[0,2,0], vertices[0,3,0]]
y = [vertices[0,0,1], vertices[0,1,1], vertices[0,2,1], vertices[0,3,1]]
plt.plot(x, y, 'b--', lw=1.4)
plt.imshow(color_select)
plt.imshow(line_image)
plt.show()


# Cany Edge Detection

gray_img = f.gray(line_image)
cany_img = f.cany_edge(gray_img, 50, 150)
# Hough Transform
# Parameters:
rho = 1
theta = np.pi/180
threshold = 1
min_line_length = 10
max_line_gap = 1

# Function -> Image aus Canny
# hough_transform(cany_img, rho, theta, threshold, min_line_lenght, max_line_gap):
lines = f.hough_transform(cany_img, rho, theta, threshold, min_line_length, max_line_gap)
line_image_blank = np.copy(line_image)*0 # creating a blank to draw lines on

# Iterate over the output "lines" and draw lines on a blank image
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image_blank,(x1,y1),(x2,y2),(255,0,0),10)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((cany_img, cany_img, cany_img)) 
# Draw the lines on the edge image
lines_edges_img = cv2.addWeighted(color_edges, 0.8, line_image_blank, 1, 0) 

#plot Images
images = [img, color_thresholds_polygon, line_image, color_select, gray_img, \
          cany_img, lines_edges_img]
titels = ['Original','Color Threshold', 'Line Image','Color Select', 'Gray - Canny', \
          'Cany Edge Detect', 'Hough Transf']
f.plot_n(images,titels,'gray')

#plt.imshow(color_thresholds_polygon, cmap='gray_r')





print('----- Main Section done ------')




# =============================================================================
#%%
# 
# os.listdir("test_images/")
# #%%
# 
# brightHSV = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
# plt.imshow(brightHSV), plt.show()
# 
# brightYCB = cv2.cvtColor(color_img,vertices cv2.COLOR_BGR2YCrCb)
# plt.imshow(brightYCB), plt.show()
# 
# # mask = cv2.inRange(color_img,rgb_threshold_,rgb_threshold)
# 
# 
# =============================================================================
# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(cany_img)   
ignore_mask_color = 255

# This time we are defining a four sided polygon to mask
imshape = shape
vertices = np.array([[(0,imshape[0]),(450, 300), (imshape[1]-450, 300), (imshape[1],imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(cany_img, mask)

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
# 
# =============================================================================

#%%
# polygon_roi(img, imshape, y_line, x_left, x_right)

img, shape = f.readIMG('test_images/solidWhiteRight.jpg')

# Cany Edge Detection
gray_img = f.gray(cany_image)
cany_img = f.cany_edge(gray_img, 50, 150)

y_line  = 300
x_left  = 400
x_right = 550
masked_edges, vertices = f.polygon_roi(img, shape, y_line, x_left, x_right)

#plot Images
images = [img,gray_img ,cany_img, masked_edges]
titels = ['img', 'gray_img', 'cany_img', 'masked_edges']
f.plot_n(images,titels,'gray')

#print(vertices.shape)
#%%

#!clear
#read Image
img, shape = f.readIMG('test_images/solidWhiteRight.jpg')

img_color_orig = np.copy(img)
color_select   = np.copy(img)
line_image     = np.copy(img)
cany_image     = np.copy(img)
line_image     = np.copy(img)

# Define color selection criteria Mache daraus ein binary 0 & 1
rgb_threshold, color_thresh_img = f.threshold(200,200,200,img_color_orig)

# Mask pixels below the threshold
color_thresholds = (img_color_orig[:,:,0] < rgb_threshold[0]) | \
                    (img_color_orig[:,:,1] < rgb_threshold[1]) | \
                    (img_color_orig[:,:,2] < rgb_threshold[2])

#==============================================================================
# ROI - Define a Triangle
left_bottom = [40, 540]
right_bottom = [900, 540]
apex = [500, 300]
ysize = shape[0]
xsize = shape[1]

# ROI Triangel Form
region_thresholds = f.tri_roi(left_bottom, right_bottom, apex, xsize, ysize)

plt.imshow(img)
plt.show()