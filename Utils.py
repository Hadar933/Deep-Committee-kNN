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

ANOMAL_DATASETS = ['mnist', 'caltech256']  # add more if needed...

if not os.path.isdir('predictions'): os.mkdir('predictions')
if not os.path.isdir('deep_activations'): os.mkdir('deep_activations')
if not os.path.isdir('shallow_activations'): os.mkdir('shallow_activations')
if not os.path.isdir('mid_activations'): os.mkdir('mid_activations')

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
    print(tabulate([['Regular (P)', len(TP), len(FP)],
                    ['Anomal  (N)', len(FN), len(TN)]],
                   headers=['Pred/Real', 'Regular (P)', 'Anomal (N)']))

    return len(TP), len(FP), len(TN), len(FN)


def find_hyperparams_and_plot_ROC():
    size = 500
    deep_range = [1.6 + diff / 100 for diff in range(size)]
    mid_range = [255 + diff / 10 for diff in range(size)]
    shal_range = [5100 + diff for diff in range(size)]
    thresholds = [{'deep': d, 'mid': m, 'shallow': s} for d, m, s
                  in zip(deep_range, mid_range, shal_range)]
    ROC_x = []
    ROC_y = []
    for t in thresholds:
        print(t)
        anomal_classifications, reg_classifications = classify_based_on_knn_distance(t)

        anomal_maj = majority_vote(anomal_classifications)
        regular_maj = majority_vote(reg_classifications)

        TP, FP, TN, FN = score_prediction(anomal_maj, regular_maj)
        P = TP + FN
        N = TN + FP
        TPR = TP / P
        FPR = FP / N

        ROC_x.append(FPR)
        ROC_y.append(TPR)

        print("===============================================")

    AUC = sum([ROC_y[i] * 1 / size for i in range(size)])

    plt.scatter(ROC_x, ROC_y, s=10)
    plt.xlabel("FPR")
    plt.title(f"ROC Curve, AUC={AUC:.5f}")
    plt.grid()
    plt.show()
