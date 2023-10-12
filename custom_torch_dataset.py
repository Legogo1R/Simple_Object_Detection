import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2

from config import *

def dataframe_from_file(file_path):
    """
    Creates pandas dataframe from YOLOv8 dataset format
    """
    path2file = os.path.join(file_path, 'labels')
    df = pd.DataFrame()
    file_list = []
    col_names=['class', 'x', 'y', 'w', 'h','img_name']

    for file in os.listdir(path2file):
        if file[-4:] == '.txt':
            file_list.append(file)

    for file_name in file_list:
        with open(os.path.join(path2file, file_name), encoding='utf-8') as txt:
            for row in txt:
                row = row[:-1]  # Remove \n
                row = row.split(" ")
                row = [x if x != '' else None for x in row]
                # print(row)
                new_df = pd.concat([pd.DataFrame([row],dtype='float'),pd.DataFrame([file_name[:-4]])],axis=1)  # Concat labels and file name
                df = pd.concat([df, new_df], axis=0)  # Append new rows

    df.columns = col_names
    return df

df_train = dataframe_from_file(TRAIN_PATH)
df_val = dataframe_from_file(VAL_PATH)
df_test = dataframe_from_file(TEST_PATH)

df_train = df_train.dropna()
df_val = df_val.dropna()
df_test = df_test.dropna()


class FruitDataset(Dataset):
    
    def __init__(self, dataframe, img_dir, mode = 'train', transforms = None):
        
        super().__init__()
        self.img_names = dataframe['img_name'].unique()
        self.dataframe = dataframe
        self.image_dir = img_dir
        self.mode = mode
        self.transforms = transforms

    def __getitem__(self, index):
        # Get records (labels) for input image
        img_name = self.img_names[index]
        records = df_train[df_train['img_name'] == img_name]

        # Get input image
        image = cv2.imread(os.path.join(self.image_dir, img_name + '.jpg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0  # Normalization

        if self.mode == 'train' or self.mode == 'valid':

            # Converting x, y, w, h to x1, y1, x2, y2
            bboxes = np.zeros((records.shape[0], 5))  # Create dummy of desired shape for targets
            bboxes[:, 0:4] = records[['x', 'y', 'w', 'h']].values
            bboxes[:, 0] = records['x'].values - records['w'].values / 2
            bboxes[:, 1] = records['y'].values - records['h'].values / 2
            bboxes[:, 2] = records['x'].values + records['w'].values / 2
            bboxes[:, 3] = records['y'].values + records['h'].values / 2
            bboxes[:, 4] = records['class'].values  ## ASK SASHA; convert to integer

            sample = {'img' : image, 'bboxes' : bboxes}

            # Aplying transforms to data if you need it
            if self.transforms:
                sample = self.transforms(sample)

        elif self.mode == 'test':

            # We just need to apply transoforms and return image
            sample = {'img' : image}

            if self.transforms:
                sample = self.transforms(sample)

        return sample
    
    def __len__(self):
        return self.img_names.shape[0]

