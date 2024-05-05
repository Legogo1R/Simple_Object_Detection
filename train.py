import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import random
from tqdm.auto import tqdm

plt.style.use('ggplot')

from config import *

from model import create_model
from utils import (
    Averager, 
    SaveBestModel, 
    save_model, 
    save_loss_plot,
    save_mAP
)
from dataset import dataframe_from_file, FruitDataset

# Set path for torch.hub weights download directory
torch.hub.set_dir(TORCH_HUB_PATH)

SEED = 666
def lock_seed(seed = SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
lock_seed()


def train(train_dataloader, model):
    """
    Function for running training itterations
    """
    print('Training the model...')
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5, last_epoch=-1, verbose=False)

    # initialize tqdm progress bar
    progress_bar = tqdm(train_dataloader, total=len(train_dataloader))

    for i, data in enumerate(progress_bar):

        optimizer.zero_grad()
        image, targets = data['img'], data['annot']
        # image.to(DEVICE)

        cls_loss, regr_loss = model(image, targets)
        print(cls_loss, regr_loss)


        










if __name__ == '__main':

    # Loading datasets and dataloaders
    dataset_train = torch.load('torch_dataset/dataset_train.pt')
    dataset_val = torch.load('torch_dataset/dataset_val.pt')
    dataset_test = torch.load('torch_dataset/dataset_test.pt')

    dataloader_train = torch.load('torch_dataset/dataloader_train.pt')
    dataloader_val = torch.load('torch_dataset/dataloader_val.pt')
    dataloader_test = torch.load('torch_dataset/dataloader_test.pt')