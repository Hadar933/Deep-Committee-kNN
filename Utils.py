import torch
from torchvision import transforms
import os
import gc
import shutil

gc.collect()
torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ResNet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
ResNet.eval()
ResNet.to(device)

batch_size = 100

ANOMAL_DATASETS = ['mnist', 'caltech101']  # add more if needed...

if not os.path.isdir('predictions'): os.mkdir('predictions')
for act in [f"{layer}_block{i}_activation" for layer in ['layer1', 'layer2', 'layer3', 'layer4'] for i in [0, 1]] + [
    'avgpool_activation']:
    if not os.path.isdir(act): os.mkdir(act)

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


def score_prediction(anomal_pred, reg_pred):
    TP = reg_pred[reg_pred]  # said T and it is T (correct reg)
    FP = reg_pred[~reg_pred]  # said F and it is T (wrong reg)
    TN = anomal_pred[~anomal_pred]  # said F and it is F (correct_anomal)
    FN = anomal_pred[anomal_pred]  # said T and it is F (wrong_anomal)
    # print(tabulate([['Regular (P)', len(TP), len(FP)],
    #                 ['Anomal  (N)', len(FN), len(TN)]],
    #                headers=['Pred/Real', 'Regular (P)', 'Anomal (N)']))

    return len(TP), len(FP), len(TN), len(FN)


def clear_dir():
    """
    deletes all directories of activations / predictions
    """
    for item in os.listdir():
        if item != 'config' and item != 'data':
            shutil.rmtree(item)

