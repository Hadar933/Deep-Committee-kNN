from typing import List
import torch
from torchvision import transforms
import os
import gc

gc.collect()
torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ResNet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
ResNet.eval()
ResNet.to(device)
batch_size = 100

if not os.path.isdir('predictions'): os.mkdir('predictions')
if not os.path.isdir('deep_activations'): os.mkdir('deep_activations')
if not os.path.isdir('shallow_activations'): os.mkdir('shallow_activations')

greyscale_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
rgb_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def emd_3d(test, train):
    """
    this is a version of earth movers distance that is performed on a shallow activation
    :param test: test batch with shape (b,c,h,w)
    :param train: train batch with the same shape
    :return:
    """
    for i in [-1, -2, -3]:
        test = torch.cumsum(test, i)
        train = torch.cumsum(train, i)
    return test, train


def imagenet_categories():
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories
