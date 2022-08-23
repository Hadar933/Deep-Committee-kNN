# %% imports
from typing import Callable
from Utils import device, ResNet, batch_size, rgb_preprocess, greyscale_preprocess, ANOMAL_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn
from torchvision import datasets
from torch.utils.data import Dataset
import os
from PIL import Image


def load_dataloaders(train_size: int, test_size: int, anomal_class: str, reg_class: str, is_one_vs_other: bool,
                     mv_net: bool, *args):
    """
    loads the relevant regular and anomalous datasets and slices the train set
    :param mv_net: true iff we want to load that specific dataset
    :param is_one_vs_other: True iff we want to use cifar10 as one vs other (one class = anomalous, the other = regular)
    :param reg_class: a string that represents the name of the regular class
    :param anomal_class: a string that represents the name of the anomalous class
    :param test_size: we perform the algorithm based on this much test samples
    :param train_size: we perform the algorithm based on this much train samples
    :param args: when is_one_vs_other == True, args[0] is an int associated with the anomal classes in one-vs-other,
    :return:
    """
    if is_one_vs_other:
        regular_target = args[0][0]
        if anomal_class == 'mnist' and reg_class == 'mnistcls':
            reg_train_data = datasets.MNIST(root='./data', train=True, download=True, transform=greyscale_preprocess)
            reg_test_data = datasets.MNIST(root='./data', train=False, download=True, transform=greyscale_preprocess)
            anomal_test_data = datasets.MNIST(root='./data', train=False, download=True, transform=greyscale_preprocess)

        elif anomal_class == 'cifar10' and reg_class == 'cifar10cls':
            reg_train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=rgb_preprocess)
            reg_test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=rgb_preprocess)
            anomal_test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=rgb_preprocess)

        else:
            raise ValueError(f"One-vs-Other does not support reg=({reg_class}), anomal=({anomal_class})")

        all_other_classes_test = anomal_test_data.data[torch.tensor(reg_test_data.targets) != regular_target]
        one_regular_class_train = reg_train_data.data[torch.tensor(reg_train_data.targets) == regular_target]
        one_regular_class_test = reg_test_data.data[torch.tensor(reg_test_data.targets) == regular_target]

        reg_train_data.data = one_regular_class_train
        reg_test_data.data = one_regular_class_test
        anomal_test_data.data = all_other_classes_test

        if len(reg_train_data.data) > train_size:
            reg_train_data = torch.utils.data.Subset(reg_train_data, range(0, train_size))
        if len(reg_test_data.data) > test_size:
            reg_test_data = torch.utils.data.Subset(reg_test_data, range(0, test_size))
        if len(anomal_test_data.data) > test_size:
            anomal_test_data = torch.utils.data.Subset(anomal_test_data, range(0, test_size))

    elif mv_net:
        # in this case we ignore input sizes and set it up manually:
        reg_mvnet = MVNetDataset(
            "G:\\My Drive\\Master\\Year 1\\Applied Deep Learning\\Deep-Committee-kNN\\transistor\\NORMAL", False)
        reg_mvnet_train_size = int(0.8 * len(reg_mvnet))
        reg_mvnet_test_size = len(reg_mvnet) - reg_mvnet_train_size

        reg_train_data, reg_test_data = torch.utils.data.random_split(reg_mvnet, [reg_mvnet_train_size, reg_mvnet_test_size])

        anomal_test_data = MVNetDataset(
            "G:\\My Drive\\Master\\Year 1\\Applied Deep Learning\\Deep-Committee-kNN\\transistor\\ANOMAL", True)
        if len(reg_train_data) > train_size:
            reg_train_data = torch.utils.data.Subset(reg_train_data, range(0, train_size))
        if len(reg_test_data) > test_size:
            reg_test_data = torch.utils.data.Subset(reg_test_data, range(0, test_size))
        if len(anomal_test_data) > test_size:
            anomal_test_data = torch.utils.data.Subset(anomal_test_data, range(0, test_size))

    else:
        if reg_class != 'cifar10':
            raise ValueError(f"Unsupported regular data set ({reg_class})")

        if reg_class == 'cifar10':
            reg_train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=rgb_preprocess)
            reg_test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=rgb_preprocess)

        if anomal_class not in ANOMAL_DATASETS:
            raise ValueError(f"Unsupported anomalous data set ({anomal_class})")

        elif anomal_class == 'mnist':
            anomal_test_data = datasets.MNIST(root='./data', train=False, download=True, transform=greyscale_preprocess)
        elif anomal_class == 'caltech101':
            anomal_test_data = datasets.Caltech101(root='./data', download=True, transform=rgb_preprocess)

        if len(reg_train_data.data) > train_size:
            reg_train_data = torch.utils.data.Subset(reg_train_data, range(0, train_size))
        if len(reg_test_data.data) > test_size:
            reg_test_data = torch.utils.data.Subset(reg_test_data, range(0, test_size))
        if len(anomal_test_data.data) > test_size:
            anomal_test_data = torch.utils.data.Subset(anomal_test_data, range(0, test_size))

    reg_train_loader = torch.utils.data.DataLoader(reg_train_data, shuffle=True, batch_size=batch_size)
    reg_test_loader = torch.utils.data.DataLoader(reg_test_data, batch_size=batch_size)
    anomal_test_loader = torch.utils.data.DataLoader(anomal_test_data, batch_size=batch_size)

    return reg_train_loader, reg_test_loader, anomal_test_loader


_anomal_activations = {}
_regular_activations = {}


def _get_activation(layer_name: str, dataset_name: str, is_anomal: bool) -> Callable:
    """
    when given as input to register_forward_hook, this function is implicitly called when model.forward() is performed
    and saves the output of layer 'name' in the dictionary described above.
    :param layer_name: name of the layer
    :param dataset_name: name of model
    :return:
    """

    def hook(model, input, output):
        if is_anomal:
            _anomal_activations[layer_name] = output.detach()

        else:
            _regular_activations[layer_name] = output.detach()

    return hook


def calculate_activations_and_save(dataloader, test_or_train: str, dataset_name: str, is_anomal) -> None:
    activations = _anomal_activations if is_anomal else _regular_activations
    anomal_suffix = "anomal" if is_anomal else "regular"

    names_and_layers_dict = {
        'layer1_block0': {'hook': ResNet.layer1[0], 'activation': torch.tensor([])},
        'layer1_block1': {'hook': ResNet.layer1[1], 'activation': torch.tensor([])},
        'layer2_block0': {'hook': ResNet.layer2[0], 'activation': torch.tensor([])},
        'layer2_block1': {'hook': ResNet.layer2[1], 'activation': torch.tensor([])},
        'layer3_block0': {'hook': ResNet.layer3[0], 'activation': torch.tensor([])},
        'layer3_block1': {'hook': ResNet.layer3[1], 'activation': torch.tensor([])},
        'layer4_block0': {'hook': ResNet.layer4[0], 'activation': torch.tensor([])},
        'layer4_block1': {'hook': ResNet.layer4[1], 'activation': torch.tensor([])},
        'avgpool': {'hook': ResNet.avgpool, 'activation': torch.tensor([])}}

    for name, layer in names_and_layers_dict.items():
        layer['hook'].register_forward_hook(_get_activation(name, dataset_name, is_anomal))
        layer['activation'] = layer['activation'].to(device)

    count = 0
    for X, _ in tqdm(dataloader):
        X = X.to(device)
        count += 1
        _ = ResNet(X)

        curr_activations = {layer_name: activations[layer_name] for layer_name in names_and_layers_dict}
        for name, layer in names_and_layers_dict.items():
            layer['activation'] = torch.cat((layer['activation'], curr_activations[name]))

        mod = 1 if dataset_name == 'mknet' else 10 # mknet is small so we save more frequent
        if count % 1 == 0:  # saving every 10 iterations (=batches) and re-initializing
            for name, layer in names_and_layers_dict.items():
                torch.save(layer['activation'],
                           f"{name}_activation/{dataset_name}_{test_or_train}_{anomal_suffix}_{count // 1}")
                layer['activation'] = torch.tensor([])
                layer['activation'] = layer['activation'].to(device)


class ActivationDataset(Dataset):
    def __init__(self, train_dataset_name: str, test_or_train: str, is_anomalous: bool) -> None:
        self.dataset_name = train_dataset_name
        self.test_or_train = test_or_train
        self.is_anomalous = is_anomalous
        self.anomal_suffix = "anomal" if is_anomalous else "regular"
        all_files = os.listdir('layer1_block0_activation')  # just to get length
        test_len = len(
            [t for t in all_files if 'test' in t and self.anomal_suffix in t])  # TODO: make sure this works
        train_len = len([t for t in all_files if 'train' in t])
        self.len = test_len if self.test_or_train == 'test' else train_len

    def __len__(self) -> int:
        return self.len - 1

    def __getitem__(self, idx: int):
        if idx >= self.len: raise IndexError(f"idx={idx}>={self.len}=len")
        activation_lst = []
        for name in [f"{layer}_block{i}" for layer in ['layer1', 'layer2', 'layer3', 'layer4'] for i in [0, 1]] + [
            'avgpool']:
            activ = torch.load(
                f"{name}_activation/{self.dataset_name}_{self.test_or_train}_{self.anomal_suffix}_{idx + 1}")
            activ = activ.to(device)
            activ = torch.nn.functional.normalize(activ)
            activation_lst.append(activ)
        return activation_lst


class MVNetDataset(Dataset):
    def __init__(self, root_dir, is_anomal):
        self.root_dir = root_dir
        self.items = os.listdir(root_dir)
        self.is_anomal = is_anomal

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        im_path = self.items[idx]
        im_obj = Image.open(f"{self.root_dir}/{im_path}")
        im_transformed = greyscale_preprocess(im_obj)
        return im_transformed, 1  # we return some redundant target 1 that is ignored just to stay consistent with other datasets
