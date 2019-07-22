import torch
import torch.nn as nn
import torch.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torchvision import models

from collections import OrderedDict
import numpy as np
import argparse
import os
from time import time

DATA_PATH = None
SAVE_PATH = None
ARCH_NET = None
CUDA = None

learning_rate = None
epochs = None
batch_size = None
hidden_units = None


# 读取数据
def read_data(train_dir, valid_dir, test_dir):
    train_data_transforms = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    other_data_transforms = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_datasets = dsets.ImageFolder(root=train_dir, transform=train_data_transforms)
    valid_datasets = dsets.ImageFolder(root=valid_dir, transform=other_data_transforms)
    test_datasets = dsets.ImageFolder(root=test_dir, transform=other_data_transforms)

    return train_datasets, valid_datasets, test_datasets


# 加载dataloder
def load_data(train_datasets, valid_datasets, test_datasets):   
    train_dataloaders = torch.utils.data.DataLoader(dataset=train_datasets,  batch_size=batch_size, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(dataset=valid_datasets,  batch_size=batch_size)
    test_dataloaders = torch.utils.data.DataLoader(dataset=test_datasets,  batch_size=batch_size)

    return train_dataloaders, valid_dataloaders, test_dataloaders


# 创建模型
def create_model(n_classes):
    model = None

    if 'vgg16' == ARCH_NET:
        model = models.vgg16(pretrained=True)
    elif 'vgg13' == ARCH_NET:
        model = models.vgg13(pretrained=True)
    else:
        print('error arch')
        return model

    for param in model.parameters():
        param.requires_grad = False
    
    layers = OrderedDict()
    layers['linear0'] = nn.Linear(7*7*512, hidden_units[0])
    layers['drop0'] = nn.Dropout(p=0.5)
    layers['relu0'] = nn.ReLU(inplace=True) 

    for i, (input_dim, output_dim) in enumerate(zip(hidden_units[1:], hidden_units[2:]), start=1):
        layers['linear'+str(i)] = nn.Linear(input_dim, output_dim)
        layers['drop'+str(i)] = nn.Dropout(p=0.5)
        layers['relu'+str(i)] = nn.ReLU(inplace=True) 

    layers['linear'+str(len(hidden_units))] = nn.Linear(hidden_units[-1], n_classes)

    model.classifier = nn.Sequential(layers)

    return model


# 训练函数
def train(model, criterion, optimizer, train_loader, valid_loader, n_val):
    loss_list = []
    accuaracy_list = []
    for epoch in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            if CUDA:
                x = x.cuda()
                y = y.cuda()
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss_list.append(loss.data.item())
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print("Epoch: {}/{}... ".format(epoch+1, epochs),
                      "Loss: {:.4f}".format(loss))

        correct = 0
        model.eval()
        for x_val, y_val in valid_loader:
            if CUDA:
                x_val = x_val.cuda()
                y_val = y_val.cuda()
            z = model(x_val)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_val).sum().item()
            
        accuaacy = correct / n_val
        accuaracy_list.append(accuaacy)
        print("Epoch: {}/{}... ".format(epoch+1, epochs), "accuarcy: {:.4f}".format(accuaacy))
        if CUDA:
            torch.cuda.empty_cache()

    return accuaracy_list, loss_list


# 保存模型和相关信息
def save_model(model, train_datasets, optimizer, n_classes):

    checkpoint = {
    'arch': ARCH_NET,
    'class_to_idx': train_datasets.class_to_idx,
    'param_state_dict': model.state_dict(),
    'optim_state_dict': optimizer.state_dict(),
    'hidden_units': hidden_units,
    'n_classes': n_classes
    }

    time_id = time()
    torch.save(checkpoint, SAVE_PATH + '/model-checkpoint-{}.pth'.format(int(time_id)))


def main():
    train_dir = DATA_PATH + '/train'
    valid_dir = DATA_PATH + '/valid'
    test_dir = DATA_PATH + '/test'

    train_datasets, valid_datasets, test_datasets = read_data(train_dir, valid_dir, test_dir)

    train_dataloaders, valid_dataloaders, _ = load_data(train_datasets, valid_datasets, test_datasets)

    n_classes = len(np.unique([train_datasets[i][1] for i in range(len(train_datasets))]))
    
    model = create_model(n_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.classifier.parameters(), momentum=0.9, lr=1e-2)
    
    if CUDA:
        model.cuda()
        criterion.cuda()
        torch.backends.cudnn.benchmark=True
        print('gpu for training...')
    else:
        print('cpu for training...')
    
    train(model, criterion, optimizer, train_dataloaders, valid_dataloaders, len(valid_datasets))

    if SAVE_PATH is not None:
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)
        save_model(model, train_datasets, optimizer, n_classes)

    print('done...')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', help='path for data', default='./flowers', type=str)
    parser.add_argument('--save_dir',help='save model path', default='./save_model', type=str)
    parser.add_argument('--arch', help='chose pre-trained network(vgg13, vgg16)', default='vgg16', type=str)
    parser.add_argument('--learning_rate', help='learning rate', default=0.01, type=float)
    parser.add_argument('--epochs', help='epochs', default=1, type=int)
    parser.add_argument('--batch_size', help='batch_size', default=128, type=int)
    parser.add_argument('--gpu', help='use gpu', default=torch.cuda.is_available(), type=bool)
    parser.add_argument('--hidden_units', help='hidden units for hidden layers', metavar='N', nargs='+', default=[512], type=int)

    args = parser.parse_args()
    DATA_PATH = args.data_directory
    SAVE_PATH = args.save_dir
    ARCH_NET = args.arch
    CUDA = args.gpu
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    hidden_units = args.hidden_units

    main()

