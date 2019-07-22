import torch
import torch.nn as nn
import torch.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torchvision import models

from PIL import Image
from collections import OrderedDict
import numpy as np
import json
import argparse
import os
from time import time

image_file = None
checkpoint_file = None
top_k = None
category_names_file = None
CUDA = None


# 预处理图像
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    w, h = im.size

    if w < h:
        im = im.resize((256, (256 * h // w)))
    else:
        im = im.resize(((256 * w // h), 256))

    center_x = w // 2
    center_y = h // 2

    box = center_x - 224//2, center_y - 224//2, center_x + 224//2, center_y + 224//2

    im = im.crop(box)

    np_image = np.array(im) / 255

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    norm_image = (np_image - means) / stds

    return norm_image.transpose((2, 0 ,1))


# 读取模型
def load_checkpoint(file_path):
    checkpoint = torch.load(file_path)
    
    if 'vgg16' == checkpoint['arch']:
        model = models.vgg16(pretrained=True)
    elif 'vgg13' == checkpoint['arch']:
        model = models.vgg13(pretrained=True)
    else:
        print('error arch')
        return None
    
    hidden_units = checkpoint['hidden_units']
    n_classes = checkpoint['n_classes']

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
    
    model.load_state_dict(checkpoint['param_state_dict'])
    
    optimizer = torch.optim.SGD(model.classifier.parameters(), momentum=0.9, lr=1e-2)
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    
    class_to_index = checkpoint['class_to_idx']
    
    return model, optimizer, class_to_index


# 预测函数
def predict(image_path, model, index_to_class):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    with torch.no_grad():
        img = process_image(image_path)
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)
        model.eval()

        if CUDA:
            model.cuda()
            img = img.cuda()
            torch.backends.cudnn.benchmark=True
            print('gpu for prediction...')
        else:
            print('cpu for prediction...')

        z = model.forward(img)
        probs, index = torch.topk(torch.softmax(z, 1), top_k)

        if CUDA:
            classes = [index_to_class[i] for i in index.cpu().data[0].numpy()]
            probs = probs.cpu().data[0].numpy().tolist()
        else:
            classes = [index_to_class[i] for i in index.data[0].numpy()]
            probs = probs.data[0].numpy().tolist()

        return probs, classes


def main():
    if not os.path.exists(image_file):
        print('image_file not existed...')
        return

    if not os.path.exists(checkpoint_file):
        print('checkpoint_file not existed...')
        return
    
    if not os.path.exists(category_names_file):
        print('category_names_file not existed...')
        return
    
    with open(category_names_file, 'r') as f:
        cat_to_name = json.load(f)
    
    model, _, class_to_index = load_checkpoint(checkpoint_file)
    
    index_to_class = {}
    for key ,value in class_to_index.items():
        index_to_class[value] = key

    probs, classes = predict(image_file, model, index_to_class)
    
    classes_to_name = [cat_to_name[key] for key in classes]

    print('top{} classes: {}'.format(top_k, classes_to_name))
    print('probs:', probs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input image', default='''./image_06743.jpg''', type=str)
    parser.add_argument('--checkpoint', help='checkpoint file', default='''./save_model/model-checkpoint-1544957216.pth''', type=str)
    parser.add_argument('--top_k', help='top_k', default=3, type=int)
    parser.add_argument('--category_names', help='json file for category_names', default='cat_to_name.json', type=str)
    parser.add_argument('--gpu', help='use gpu', default=torch.cuda.is_available(), type=bool)
 
    args = parser.parse_args()
    image_file = args.input
    checkpoint_file = args.checkpoint
    top_k = args.top_k
    category_names_file = args.category_names
    CUDA = args.gpu

    main()
 