import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, CALTECH_DATA_FOLDER, CALTECH_CLASSES, BATCH_SIZE
from datasets import Caltech101Loader


def train_phase(train_loader=None, model=None, criterion=None, optimizer=None):

    n_iterations = len(train_loader)
    size = 0
    total_loss = 0
    accuracy = 0

    for i, (images, labels) in tqdm(enumerate(train_loader), total=n_iterations):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        size += labels.size(0)

        outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss += loss.item()

        y_hat = F.softmax(outputs, dim=1)
        value_softmax, index_softmax = torch.max(y_hat.data, 1)
        accuracy += (labels == index_softmax).sum().item()

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = total_loss / n_iterations
    train_accuracy = accuracy / size
    return train_loss, train_accuracy*100


def test_phase(test_loader=None, model=None, criterion=None):

    n_iterations = len(test_loader)
    size = 0
    total_loss = 0
    accuracy = 0

    for i, (images, labels) in tqdm(enumerate(test_loader), total=n_iterations):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        size += labels.size(0)

        outputs = model(images)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        y_hat = F.softmax(outputs, dim=1)
        value_softmax, index_softmax = torch.max(y_hat.data, 1)
        accuracy += (labels == index_softmax).sum().item()

    test_loss = total_loss / n_iterations
    test_accuracy = accuracy / size
    return test_loss, test_accuracy*100


# Single-size training
def run_train(train_model, criterion, optimizer):
    f = open("history.txt", 'a')
    
    dataloader = Caltech101Loader(data_folder=CALTECH_DATA_FOLDER, phase="train", crop_dim=224, batch_size=BATCH_SIZE)
    train_loader, test_loader = dataloader.load_data()

    loss_opt = 1e9
    acc_opt = -1e9

    history_train_loss = []
    history_test_loss = []
    history_train_acc = []
    history_test_acc = []

    for epoch in range(NUM_EPOCHS):
        print()
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]:')

        train_loss, train_accuracy = train_phase(train_loader=train_loader, model=train_model, criterion=criterion, optimizer=optimizer)
        print('Train Loss: {:.4f}, Train Accuracy: {:.4f}%' .format(train_loss, train_accuracy))

        test_loss, test_accuracy = test_phase(test_loader=test_loader, model=train_model, criterion=criterion)
        print('Valid Loss: {:.4f}, Valid Accuracy: {:.4f}%' .format(test_loss, test_accuracy))
        
        f.write(f"===== Epoch{epoch + 1}:\n")
        f.write(f"{train_loss} {train_accuracy} {test_loss} {test_accuracy}\n\n")

        history_train_loss.append(train_loss)
        history_test_loss.append(test_loss)
        history_train_acc.append(train_accuracy)
        history_test_acc.append(test_accuracy)

        if test_loss < loss_opt and test_accuracy >= acc_opt:
            loss_opt = test_loss
            acc_opt = test_accuracy
            torch.save(train_model.state_dict(), 'zf5_best_2.pt')
        torch.save(train_model.state_dict(), 'zf5_last_2.pt')
    return history_train_loss, history_test_loss, history_train_acc, history_test_acc


# Multi-size training
def run_train_multi_size(train_model_1, train_model_2, criterion, optimizer_1, optimizer_2):
    f = open("history.txt", 'a')
    
    dataloader_1 = Caltech101Loader(data_folder=CALTECH_DATA_FOLDER, phase="train", crop_dim=224, batch_size=BATCH_SIZE)
    train_loader_1, test_loader_1 = dataloader_1.load_data()
    
    dataloader_2 = Caltech101Loader(data_folder=CALTECH_DATA_FOLDER, phase="train", crop_dim=180, batch_size=BATCH_SIZE)
    train_loader_2, test_loader_2 = dataloader_2.load_data()
    
    loss_opt = 1e9
    acc_opt = -1e9

    history_train_loss = []
    history_test_loss = []
    history_train_acc = []
    history_test_acc = []
    
    
    for epoch in range(NUM_EPOCHS):
        print()
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]:')
        
        train_loss = 0.
        train_accuracy = 0.
        test_loss = 0.
        test_accuracy = 0.
        
        if epoch % 2 == 0:
            train_loss, train_accuracy = train_phase(train_loader=train_loader_1, model=train_model_1, criterion=criterion, optimizer=optimizer_1)
            print('Train Loss: {:.4f}, Train Accuracy: {:.4f}%' .format(train_loss, train_accuracy))

            test_loss, test_accuracy = test_phase(test_loader=test_loader_1, model=train_model_1, criterion=criterion)
            print('Valid Loss: {:.4f}, Valid Accuracy: {:.4f}%' .format(test_loss, test_accuracy))
        else:
            train_model_2.load_state_dict(train_model_1.state_dict())
            
            train_loss, train_accuracy = train_phase(train_loader=train_loader_2, model=train_model_2, criterion=criterion, optimizer=optimizer_2)
            print('Train Loss: {:.4f}, Train Accuracy: {:.4f}%' .format(train_loss, train_accuracy))
            
            train_model_1.load_state_dict(train_model_2.state_dict())

            test_loss, test_accuracy = test_phase(test_loader=test_loader_1, model=train_model_1, criterion=criterion)
            print('Valid Loss: {:.4f}, Valid Accuracy: {:.4f}%' .format(test_loss, test_accuracy))
        
        f.write(f"===== Epoch{epoch + 1}:\n")
        f.write(f"{train_loss} {train_accuracy} {test_loss} {test_accuracy}\n\n")

        history_train_loss.append(train_loss)
        history_test_loss.append(test_loss)
        history_train_acc.append(train_accuracy)
        history_test_acc.append(test_accuracy)

        if test_loss < loss_opt and test_accuracy >= acc_opt:
            loss_opt = test_loss
            acc_opt = test_accuracy
            torch.save(train_model_1.state_dict(), 'zf5spp_multi_best.pt')
        torch.save(train_model_1.state_dict(), 'zf5spp_multi_last.pt')
    return history_train_loss, history_test_loss, history_train_acc, history_test_acc


def plot_his(his_loss_train, his_loss_test, his_acc_train, his_acc_test, n_epochs, name_save):
    x = range(1, n_epochs+1)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    axes[0].plot(x, his_loss_train, c='b')
    axes[0].plot(x, his_loss_test, c='r')
    axes[0].set_title("Loss")
    axes[0].set_xlabel("n_epochs")
    axes[1].plot(x, his_acc_train, c='b')
    axes[1].plot(x, his_acc_test, c='r')
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("n_epochs")
    axes[1].legend(["Train", "Test"])
    plt.savefig(name_save)
    plt.show()
