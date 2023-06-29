import os

import torch
import torchvision
import torch.nn as nn
import argparse

from model import ZF5NetNoSPP, ZF5NetSPP
from config import DEVICE, NUM_CLASSES, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS
from utils import run_train, run_train_multi_size, plot_his


parser = argparse.ArgumentParser()
parser.add_argument('--spp', type=bool, default=False, help="use pyramid pool or not")
parser.add_argument('--mode', type=str, default='single', help="single-size or multi-size training")
args = parser.parse_args()

if __name__ == '__main__':
    history_train_loss, history_test_loss, history_train_acc, history_test_acc = None, None, None, None
    if args.mode == "single":
        train_model = None
        if args.spp:
            train_model = ZF5NetSPP(input_channels=3, crop_dim=224, num_classes=NUM_CLASSES)
            
        else:
            train_model = ZF5NetNoSPP(input_channels=3, num_classes=NUM_CLASSES)
        train_model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(train_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        history_train_loss, history_test_loss, history_train_acc, history_test_acc = run_train(train_model=train_model, criterion=criterion, optimizer=optimizer)
    else:
        train_model_1 = ZF5NetSPP(input_channels=3, crop_dim=224, num_classes=101)
        train_model_2 = ZF5NetSPP(input_channels=3, crop_dim=180, num_classes=101)

        train_model_1.to(DEVICE)
        train_model_2.to(DEVICE)

        train_model_2.load_state_dict(train_model_1.state_dict())

        criterion = nn.CrossEntropyLoss()
        optimizer_1 = torch.optim.Adam(train_model_1.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        optimizer_2 = torch.optim.Adam(train_model_2.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        history_train_loss, history_test_loss, history_train_acc, history_test_acc = run_train_multi_size(train_model_1=train_model_1, train_model_2=train_model_2, criterion=criterion, optimizer_1=optimizer_1, optimizer_2=optimizer_2)

    plot_his(history_train_loss, history_test_loss, history_train_acc, history_test_acc, NUM_EPOCHS, "history.png")
