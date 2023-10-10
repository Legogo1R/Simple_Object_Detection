import torch
import os

BASE_PATH = "dataset"
TRAIN_PATH = os.path.sep.join([BASE_PATH, "train"])
VAL_PATH = os.path.sep.join([BASE_PATH, "valid"])
TEST_PATH = os.path.sep.join([BASE_PATH, "test"])

BASE_OUTPUT = "output"

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.pth"])
LE_PATH = os.path.sep.join([BASE_OUTPUT, "le.pickle"])  # Label encoder
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# specify ImageNet mean and standard deviation channel-wise, width-wise, and height-wise
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32

# specify the loss weights
LABELS = 1.0
BBOX = 1.0

# class mapping
CLS_MAP_DICT = {
    'apple' : 0,
    'banana' : 1,
    'grape' : 2,
    'orange' : 3,
    'pineapple' : 4,
    'water mellon' : 5,
}