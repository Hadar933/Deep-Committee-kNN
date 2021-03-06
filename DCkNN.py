import os

import scipy.spatial.distance
import torch
from tqdm import tqdm
from Data_Preprocess import ActivationDataset
from typing import Dict, Tuple, List
from Utils import device, emd_3d


def kNN(train: torch.Tensor,
        test: torch.Tensor,
        k: int,
        is_shallow: bool) -> torch.Tensor:
    """
    for a given test batch, finds the pairwise knn for every sample in the given train batch
    :param train: a batch from the train set
    :param test: a batch from the test set
    :param k: neighbours to consider (hyper parameter)
    :param is_shallow:
    :return: pairwise k-nearest-neighbours
    """
    if is_shallow:  # shallow layers have shape (c,h,w) so we need a different metric for distance
        test, train = emd_3d(test, train)
    dist = torch.cdist(test.flatten(1), train.flatten(1))
    knn = dist.topk(k, largest=False)
    return knn


def _calc_current_knn_and_append(curr_train: torch.Tensor,
                                 curr_test: torch.Tensor,
                                 dict_to_update: Dict[str, torch.Tensor],
                                 k: int,
                                 is_shallow: bool) -> None:
    """
    for a given train batch and test batch (both of specific type - deep/shallow), calculated the pairwise kNN
    and appends both values and indices to a provided return tensor
    """
    curr_knn = kNN(curr_train, curr_test, k, is_shallow)
    values, indices = curr_knn.values, curr_knn.indices
    dict_to_update['values'] = torch.cat((dict_to_update['values'], values), 1)
    dict_to_update['indices'] = torch.cat((dict_to_update['indices'], indices), 1)


def _get_predictions(knn_results: Dict[str, torch.Tensor],
                     k: int
                     ) -> torch.Tensor:
    """
    takes the final results of the knn process and returns predictions based on majority vote.
    :param k: neighbours to consider (hyperparameter)
    :param knn_results: shaped (b,k) input tensor of k-closest train labels for every test sample in the batch
    :return: majority vote tensor (b,)
    """
    final_topk = knn_results['values'].topk(k, largest=False)
    # predictions = knn_results['labels'].gather(dim=1, index=final_topk.indices)
    # majority_vote = torch.mode(predictions).values
    return final_topk.values.mean(axis=1)


def committee_kNN_from_all_files(k: int,
                                 train: ActivationDataset,
                                 test: ActivationDataset,
                                 dataset: str
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :return: classifications of the test data
    """
    shallow_ret, deep_ret = torch.tensor([], device=device), torch.tensor([], device=device)
    for curr_test_shallow, curr_test_deep in tqdm(test):  # TODO: maybe iterate both reg & anomal using zip(..)
        curr_test_shallow = curr_test_shallow.to(device)
        curr_test_deep = curr_test_deep.to(device)
        knn_shallow = {'values': torch.tensor([], device=device), 'indices': torch.tensor([], device=device)}
        knn_deep = {'values': torch.tensor([], device=device), 'indices': torch.tensor([], device=device)}

        for curr_train_shallow, curr_train_deep in tqdm(train):
            curr_train_shallow = curr_train_shallow.to(device)
            curr_train_deep = curr_train_deep.to(device)
            _calc_current_knn_and_append(curr_train_shallow, curr_test_shallow, knn_shallow, k, True)
            _calc_current_knn_and_append(curr_train_deep, curr_test_deep, knn_deep, k, False)

        shallow_ret = torch.cat((shallow_ret, _get_predictions(knn_shallow, k)))
        deep_ret = torch.cat((deep_ret, _get_predictions(knn_deep, k)))
    torch.save(deep_ret, f'predictions/deep_{dataset}')
    torch.save(shallow_ret, f'predictions/shallow_{dataset}')
    return shallow_ret, deep_ret


def load_predictions():
    """
    loads the already calculated predictions and calculates relevant measurements
    :return: a nested dict of the form {ft/der/shallow/deep: {prediction: ,classification: ,accuracy:},... }
    """
    data = {}
    deep_cifar10 = torch.load('predictions/deep_cifar10')
    shallow_cifar10 = torch.load('predictions/shallow_cifar10')
    shallow_mnist = torch.load('predictions/shallow_mnist')
    deep_mnist = torch.load('predictions/deep_mnist')
    return deep_cifar10, shallow_cifar10, shallow_mnist, deep_mnist


def find_best_thresholds():
    # TODO
    return {'deep': 6.47, 'shallow': 5000}


def committee_all_predictions():
    # loading predictions:
    thresholds = find_best_thresholds()
    predictions = {}
    files = os.listdir('predictions')
    for filename in files:
        activation = torch.load(f"predictions/{filename}")
        layer, dataset = filename.split("_")
        thres = thresholds[layer]
        pred = activation <= thres  # False iff Anomalous
        predictions[filename] = pred
