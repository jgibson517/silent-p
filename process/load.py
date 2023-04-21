import os
import pandas as pd
from torchvision.io import read_image

def collect_image_files(split):
    '''
    Aggegrates file paths into csv file containing the file handle,
    shape and size, channel and label

    Inputs:
        split (str): test, train, or val

    Outputs: csv file
    '''
    data = []
    img_dir = f"data/{split}"

    for label in ["NORMAL", "PNEUMONIA"]:

        img_dir = f"data/{split}/{label}"

        paths = os.listdir(img_dir)

        temp_df = pd.DataFrame({'path' : paths,'channel': None, 
                                'height' : None, 'width' : None, 'label' : label}) 

        for image_file in paths:
          
          full_path = label + '/' + image_file 
          channel, height, width = read_image(f'{img_dir}/{image_file}').shape
          
          temp_df.loc[(temp_df['path'] == image_file), 'channel'] = channel
          temp_df.loc[(temp_df['path'] == image_file), 'height'] = height
          temp_df.loc[(temp_df['path'] == image_file), 'width'] = width
          temp_df.loc[(temp_df['path'] == image_file), 'path'] = full_path

        data.append(temp_df)

    comb_df = pd.concat(data)

    comb_df.to_csv(f'data/output/{split}.csv')

            


