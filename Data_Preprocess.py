# %% imports
from typing import Callable
from Utils import device, ResNet, batch_size, rgb_preprocess, greyscale_preprocess
from tqdm import tqdm
import torch.nn
from torchvision import datasets
from torch.utils.data import Dataset
import torch.nn


# %% loading ResNet and ImageNet classes

def load_dataloaders(train_size: int):
    """
    loads the relevant regular and anomalous datasets and slices the train set
    :param train_size: size of the data you need from the train (for test we take everything)
    :return:
    """

    cifar10_train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=rgb_preprocess)
    cifar10_test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=rgb_preprocess)
    mnist_test_data = datasets.MNIST(root='./data', train=False, download=True, transform=greyscale_preprocess)

    cifar10_sliced_train_data = torch.utils.data.Subset(cifar10_train_data, range(0, train_size))

    cifar10_train_loader = torch.utils.data.DataLoader(cifar10_sliced_train_data, shuffle=True, batch_size=batch_size)
    cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test_data, batch_size=batch_size)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test_data, batch_size=batch_size)

    return cifar10_train_loader, cifar10_test_loader, mnist_test_loader


_mnist_activations = {}
_cifar10_activations = {}


def _get_activation(layer_name: str, dataset_name: str) -> Callable:
    """
    when given as input to register_forward_hook, this function is implicitly called when model.forward() is performed
    and saves the output of layer 'name' in the dictionary described above.
    :param layer_name: name of the layer
    :param dataset_name: name of model
    :return:
    """

    def hook(model, input, output):
        if dataset_name == 'mnist':
            _mnist_activations[layer_name] = output.detach()
        elif dataset_name == 'cifar10':
            _cifar10_activations[layer_name] = output.detach()

    return hook


def calculate_activations_and_save(dataloader, test_or_train: str, dataset_name: str) -> None:
    """
    same as calculate_activations_and_save but for all the classes together (not splitted)
    :return:
    """
    activations = _mnist_activations if dataset_name == 'mnist' else _cifar10_activations
    deep_layer_name, shallow_layer_name = 'deep_conv1', 'shallow_conv2'
    deep_hook = ResNet.layer4[1].register_forward_hook(_get_activation(deep_layer_name, dataset_name))
    shallow_hook = ResNet.layer2[1].register_forward_hook(_get_activation(shallow_layer_name, dataset_name))
    final_shallow, final_deep = torch.tensor([]), torch.tensor([])
    count = 0
    for X, _ in tqdm(dataloader):
        X = X.to(device)
        count += 1
        out = ResNet(X)
        curr_shallow_act, curr_deep_act = activations[shallow_layer_name], activations[deep_layer_name]
        final_deep = torch.cat((final_deep, curr_deep_act))
        final_shallow = torch.cat((final_shallow, curr_shallow_act))
        if count % 10 == 0:  # saving every 10 iterations (=batches) and re-initializing
            torch.save(final_deep, f'deep_activations/{dataset_name}_{test_or_train}_{count // 10}')
            torch.save(final_shallow, f'shallow_activations/{dataset_name}_{test_or_train}_{count // 10}')
            final_shallow, final_deep = torch.tensor([]), torch.tensor([])
    deep_hook.remove()
    shallow_hook.remove()


class ActivationDataset(Dataset):
    def __init__(self, train_dataset_name: str, test_or_train: str) -> None:
        self.dataset_name = train_dataset_name
        self.test_or_train = test_or_train
        self.len = 4 if self.test_or_train == 'train' else 9  # TODO make variable

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int):
        if idx > self.len: raise IndexError(f"idx={idx}>={self.len}=len")
        deep_act = torch.load(f"deep_activations/{self.dataset_name}_{self.test_or_train}_{idx + 1}")
        deep_act = deep_act.to(device)
        shallow_act = torch.load(f"shallow_activations/{self.dataset_name}_{self.test_or_train}_{idx + 1}")
        shallow_act = shallow_act.to(device)
        return torch.nn.functional.normalize(shallow_act), torch.nn.functional.normalize(deep_act)
