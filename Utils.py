from typing import List

import torch
from torchvision import transforms
import os

# initializing device and model:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ResNet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
ResNet.eval()
ResNet.to(device)
batch_size = 100

# creating relevant directories:
for dirname in ['predictions', 'deep_activations', 'shallow_activations']:
    if not os.path.isdir(dirname): os.mkdir(dirname)

# relevant transforms to perform on the dataset:
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


def imagenet_categories() -> List:
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories
