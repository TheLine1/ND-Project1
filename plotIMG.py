#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:53:15 2019

@author: linux
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_image(img, title='', cmap='', figsize=(12, 7)):
    plt.imshow(img, cmap=cmap)
    plt.title(title,fontsize=25)
    plt.show()

def plot_n(img, title, cmap=''):
    plt.figure(figsize=(18,14))
    #plt.figure(figsize=(9,7))
    plt.subplots_adjust(wspace=0.01, hspace=0.15)
        
    if len(img) % 2 > 0:
        raw = (len(img)-1) / 2 + 1
        print('raw =' + str(raw))
    else:
        raw = len(img) / 2
        print('raw =' + str(raw))
        
    for i in range(len(img)):

        plt.subplot(raw, 2 , i+1)
        plt.imshow(img[i],cmap=str(cmap))
        plt.title(title[i], fontsize = 20)
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
    
#p.plot_dual(img_orig,color_img,'orig','rgb_thres', 'gray')
def plot_dual(img1, img2, title1='', title2='', cmap='', figsize=(10.6, 6)):
    """Plot two images side by side.
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=15)
    ax2.imshow(img2, cmap=str(cmap))
    ax2.set_title(title2, fontsize=15)
    return

def plot_triple(img1, img2, img3, title1='', title2='', title3='', cmap='', figsize=(15, 9)):
    """Plot 3 images side by side.
    """
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    f.tight_layout()
    ax1.imshow(img1, cmap=str(cmap))
    ax1.set_title(title1, fontsize=15)
    ax2.imshow(img2, cmap=str(cmap))
    ax2.set_title(title2, fontsize=15)
    ax3.imshow(img3, cmap=str(cmap))
    ax3.set_title(title3, fontsize=15)
    return