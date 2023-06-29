import os
import torch
import torchvision

if not os.path.exists("./Caltech101"):
    caltech_data = torchvision.datasets.Caltech101(root='./Caltech101', target_type='category', transform=None, download=True)

CALTECH_DATA_FOLDER = "./Caltech101/caltech101/101_ObjectCategories"
CALTECH_CLASSES = os.listdir(CALTECH_DATA_FOLDER)
CALTECH_CLASSES.sort()
CALTECH_CLASSES = CALTECH_CLASSES[1:]

# Hyperparameters
NUM_CLASSES = 101
NUM_EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')