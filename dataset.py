import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
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
		records = self.dataframe[self.dataframe['img_name'] == img_name]

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

			sample = {'img' : image, 'annot' : bboxes}

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

if __name__ == '__main__':
	### Preparing Datasets and Dataloaders for Training 
	# Create pandas dataframe
	df_train = dataframe_from_file(TRAIN_PATH)
	df_val = dataframe_from_file(VAL_PATH)
	df_test = dataframe_from_file(TEST_PATH)
	# Drop None values
	df_train = df_train.dropna()
	df_val = df_val.dropna()
	df_test = df_test.dropna()
	# Pathes to images
	train_img_path = os.path.join(TRAIN_PATH, 'images')
	val_img_path = os.path.join(VAL_PATH, 'images')
	test_img_path = os.path.join(TEST_PATH, 'images')
	# Create torch datasets
	dataset_train = FruitDataset(df_train, train_img_path, mode='train')
	dataset_val = FruitDataset(df_val, val_img_path, mode='valid')
	dataset_test = FruitDataset(df_test, test_img_path, mode='test')
	# Dataloaders
	dataloader_train = DataLoader(
		dataset_train,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=4
	)
	dataloader_val = DataLoader(
		dataset_val,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=4
	)
	dataloader_test = DataLoader(
		dataset_test,
		batch_size=1,
		shuffle=True,
		num_workers=4
	)

	# Save to avoid recreating datasets in other files
	torch.save(dataset_train, 'torch_dataset/dataset_train.pt')
	torch.save(dataset_val, 'torch_dataset/dataset_val.pt')
	torch.save(dataset_test, 'torch_dataset/dataset_test.pt')

	torch.save(dataloader_train, 'torch_dataset/dataloader_train.pt')
	torch.save(dataloader_val, 'torch_dataset/dataloader_val.pt')
	torch.save(dataloader_test, 'torch_dataset/dataloader_test.pt')