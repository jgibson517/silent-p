import os
import pandas as pd

from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.io import read_image
import torch
import warnings
import cv2
import numpy as np


# Heleper Functions
def normalize(image):
    
    transforms = T.Normalize(mean=0, std=255)
    image = transforms(image.float())

    #image = image.numpy()

    # transforms = T.ToTensor()
    # image = transforms(image)
    transforms = T.Grayscale(num_output_channels=1)
    image = transforms(image)

    return image

#Sobel 
def sobel(image):

    # convert to cv2 image object
    image_np = image.numpy()
    cv2_image = np.transpose(image_np, (1, 2, 0))

    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # sobel transformation
    image = cv2.Sobel(src=cv2_image, 
                      ddepth=cv2.CV_16S,
                      dst=cv2.CV_64F,
                      dx=1, 
                      dy=1,
                      ksize=3)
    

    transforms = T.ToTensor()
    image = transforms(image)

    to_gray = T.Grayscale(num_output_channels=1)
    image = to_gray(image)
    return image

#Color function
def tophat(image):

    # convert to cv2 image object
    image_np = image.numpy()
    cv2_image = np.transpose(image_np, (1, 2, 0))

    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

    # Top Hat Transform
    topHat = cv2.morphologyEx(cv2_image, cv2.MORPH_TOPHAT, kernel)
    # Black Hat Transform
    blackHat = cv2.morphologyEx(cv2_image, cv2.MORPH_BLACKHAT, kernel)

    adj = cv2_image + topHat - blackHat

    transforms = T.ToTensor()
    image = transforms(adj)

    to_gray = T.Grayscale(num_output_channels=1)
    image = to_gray(image)

    return image


# Transformations
base_transforms = T.Compose(
    [
    # centercrop to consistent aspect ratio - relative height/width ratio
    # crop is randomizing by default
    T.CenterCrop(size=(720,720)),

    # resize - about the amount of pixels
    T.Resize((256,256)),

    # Some images read in with three channels
    T.Grayscale(num_output_channels=1)

    ])
   
edges_transforms = T.Compose(
    [
    #base
    T.CenterCrop(size=(720,720)),
    T.Resize((256,256)),
    T.Grayscale(num_output_channels=1),

    #edges
    T.GaussianBlur((5,5)),
    #T.Lambda(sobel)
    T.ColorJitter(contrast=(0,10))
    ])

color_transforms = T.Compose(
    [
    #base
    T.CenterCrop(size=(720,720)),
    T.Resize((256,256)),
    T.Grayscale(num_output_channels=1),


    #color
    #T.ToTensor(), #rescale
    normalize,
    T.Lambda(tophat)
    ])

both_transforms = T.Compose(
    [
    #base
    T.CenterCrop(size=(720, 720)),
    T.Resize((256,256)),
    T.Grayscale(num_output_channels=1),

    #edges
    T.GaussianBlur((5,5)),
    T.Lambda(sobel),
    
    #color
    T.ToTensor(), #rescale
    T.Normalize(mean=0, std=255),
    T.ColorJitter()

    ])