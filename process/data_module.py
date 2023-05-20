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

warnings.simplefilter("ignore")


     


class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir_path, transform=None):
        """
        You can set your custom dataset to take in more parameters than specified
        here. But, I recommend at least you start with the three I listed here,
        as these are standard

        csv_file (str): file path to the csv file you created

        img_dir_path: directory path to your images
        transform: Compose (a PyTorch Class) that strings together several
          transform functions (e.g. data augmentation steps)

        One thing to note -- you technically could implement `transform` within
        the dataset. No one is going to stop you, but you can think of the
        transformations/augmentations you do as a hyperparameter. If you treat
        it as a hyperparameter, you want to be able to experiment with different
        transformations, and therefore, it would make more sense to decide those
        transformations outside the dataset class and pass it to the dataset!
        """
        self.img_df = pd.read_csv(csv_file)
        self.img_dir = img_dir_path
        self.transform = transform

    def __len__(self):
        """
        Returns: (int) length of your dataset
        """
        return len(self.img_df)

    def __getitem__(self, idx):
        """
        Loads and returns your sample (the image and the label) at the
        specified index

        Parameter: idx (int): index of interest
        Returns: image, label
        """

        img_path = f'{self.img_dir}{self.img_df.loc[:,"path"][idx]}'
    
        image = read_image(img_path)
        
        label = self.img_df.loc[:,"label"][idx]
        
        # Jack: not sure if this how to set the labels, but this matches the size 
        # of the output from the model... 
        if label == 'NORMAL':
            num_label = 0
        else:
            num_label = 1

        # if you are transforming your image (i.e. you're dealing with training data),
        # you would do that here!
        if self.transform:
            image = self.transform(image)

        return image, num_label

#Sobel Helper Function 
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



# Transformations
base_transforms = T.Compose(
    [
    # centercrop to consistent aspect ratio - relative height/width ratio
    # crop is randomizing by default
    T.CenterCrop(size=(512,924)),

    # resize - about the amount of pixels
    T.Resize((256,256)),

    # Some images read in with three channels
    T.Grayscale(num_output_channels=1)

    ])

            
edges_transforms = T.Compose(
    [
    #base
    T.CenterCrop(size=(512,924)),
    T.Resize((256,256)),
    T.Grayscale(num_output_channels=1),

    #edges
    T.GaussianBlur((5,5)),
    T.Lambda(sobel)
    ])


def normalize(image):
    
    transforms = T.Normalize(mean=0, std=255)
    image = transforms(image.float())

    #image = image.numpy()

    # transforms = T.ToTensor()
    # image = transforms(image)
    transforms = T.Grayscale(num_output_channels=1)
    image = transforms(image)

    return image




color_transforms = T.Compose(
    [
    #base
    T.CenterCrop(size=(512,924)),
    T.Resize((256,256)),
    T.Grayscale(num_output_channels=1),

    #color
    #T.ToTensor(), #rescale
    normalize,
    T.ColorJitter()
    ])

both_transforms = T.Compose(
    [
    #base
    T.CenterCrop(size=(512,924)),
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



            

    


