import os
import pandas as pd

from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.io import read_image



class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir_path, transform=None):
        """
        You can set your custom dataset to take in more parameters than specified
        here. But, I recommend at least you start with the three I listed here,
        as these are standard

        csv_file (str): file path to the csv file you created /
        df (pandas df): pandas dataframe

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
        self.img_labels = pd.read_csv(csv_file, usecols=['label'])
        self.img_dir = img_dir_path
        self.transform = transform

    def __len__(self):
        """
        Returns: (int) length of your dataset
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Loads and returns your sample (the image and the label) at the
        specified index

        Parameter: idx (int): index of interest

        Returns: image, label
        """

        img_path = None
        
        image = None

        label_idx = None
        
        label = self.img_labels.iloc[idx, label_idx]

        # if you are transforming your image (i.e. you're dealing with training data),
        # you would do that here!
        if self.transform:
            image = self.transform(image)

        return image, label



transforms = T.Compose(
    [
        T.RandomAdjustSharpness(sharpness_factor=2),
        T.RandomPosterize(bits=4, p=0.5)
    ] )


training_data = CustomImageDataset(csv, img_dir_path, transforms)
val_data = CustomImageDataset(csv, img_dir_path, transforms)
test_data = CustomImageDataset(csv, img_dir_path, transforms)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


            


