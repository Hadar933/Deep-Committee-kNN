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


# def cumsum_3d(a):
#     a = torch.cumsum(a, -1)
#     a = torch.cumsum(a, -2)
#     a = torch.cumsum(a, -3)
#     return a
#
#
# def norm_3d(a):
#     return a / torch.sum(a, dim=(-1, -2, -3), keepdim=True)
#

def emd_3d(a, b):
    # a = norm_3d(a)
    # b = norm_3d(b)
    for i in [-1, -2, -3]:
        a = torch.cumsum(a, i)
        b = torch.cumsum(b, i)
    # return torch.mean(torch.square(a - b), dim=(-1, -2, -3))
    return a,b

def imagenet_categories():
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories
