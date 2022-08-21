# %% imports
from typing import Callable
from Utils import device, ResNet, batch_size, rgb_preprocess, greyscale_preprocess, ANOMAL_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn
from torchvision import datasets
from torch.utils.data import Dataset
import os


def load_dataloaders(train_size: int, test_size: int, anomal_class: str, reg_class: str, is_one_vs_other: bool, *args):
    """
    loads the relevant regular and anomalous datasets and slices the train set
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

    deep_layer_name, mid_layer_name, shallow_layer_name = 'deep_conv1', 'mid_conv1', 'shallow_conv2'

    deep_hook = ResNet.avgpool.register_forward_hook(_get_activation(deep_layer_name, dataset_name, is_anomal))
    mid_hook = ResNet.layer4[1].register_forward_hook(_get_activation(mid_layer_name, dataset_name, is_anomal))
    shallow_hook = ResNet.layer4[0].register_forward_hook(_get_activation(shallow_layer_name, dataset_name, is_anomal))

    final_shallow, final_mid, final_deep = torch.tensor([]), torch.tensor([]), torch.tensor([])
    final_deep = final_deep.to(device)
    final_shallow = final_shallow.to(device)
    final_mid = final_mid.to(device)

    count = 0
    for X, _ in tqdm(dataloader):
        X = X.to(device)
        count += 1
        _ = ResNet(X)
        curr_shallow_act, curr_mid_act, curr_deep_act = activations[shallow_layer_name], activations[mid_layer_name], \
                                                        activations[deep_layer_name]
        final_deep = torch.cat((final_deep, curr_deep_act))
        final_mid = torch.cat((final_mid, curr_mid_act))
        final_shallow = torch.cat((final_shallow, curr_shallow_act))
        if count % 10 == 0:  # saving every 10 iterations (=batches) and re-initializing
            torch.save(final_deep, f'deep_activations/{dataset_name}_{test_or_train}_{anomal_suffix}_{count // 10}')
            torch.save(final_mid, f'mid_activations/{dataset_name}_{test_or_train}_{anomal_suffix}_{count // 10}')
            torch.save(final_shallow,
                       f'shallow_activations/{dataset_name}_{test_or_train}_{anomal_suffix}_{count // 10}')
            final_shallow, final_mid, final_deep = torch.tensor([]), torch.tensor([]), torch.tensor([])
            final_deep = final_deep.to(device)
            final_mid = final_mid.to(device)
            final_shallow = final_shallow.to(device)
    deep_hook.remove()
    mid_hook.remove()
    shallow_hook.remove()


class ActivationDataset(Dataset):
    def __init__(self, train_dataset_name: str, test_or_train: str, is_anomalous: bool) -> None:
        self.dataset_name = train_dataset_name
        self.test_or_train = test_or_train
        self.is_anomalous = is_anomalous
        self.anomal_suffix = "anomal" if is_anomalous else "regular"
        all_files = os.listdir('deep_activations')  # just to get length
        test_len = len(
            [t for t in all_files if 'test' in t and self.anomal_suffix in t])  # TODO: make sure this works
        train_len = len([t for t in all_files if 'train' in t])
        self.len = test_len if self.test_or_train == 'test' else train_len

    def __len__(self) -> int:
        return self.len - 1

    def __getitem__(self, idx: int):
        if idx >= self.len: raise IndexError(f"idx={idx}>={self.len}=len")
        deep_act = torch.load(
            f"deep_activations/{self.dataset_name}_{self.test_or_train}_{self.anomal_suffix}_{idx + 1}")
        deep_act = deep_act.to(device)
        mid_act = torch.load(f"mid_activations/{self.dataset_name}_{self.test_or_train}_{self.anomal_suffix}_{idx + 1}")
        mid_act = mid_act.to(device)
        shallow_act = torch.load(
            f"shallow_activations/{self.dataset_name}_{self.test_or_train}_{self.anomal_suffix}_{idx + 1}")
        shallow_act = shallow_act.to(device)
        return torch.nn.functional.normalize(shallow_act), torch.nn.functional.normalize(
            mid_act), torch.nn.functional.normalize(deep_act)