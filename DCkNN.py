import os
import torch
from tqdm import tqdm
from Data_Preprocess import ActivationDataset, calculate_activations_and_save
from typing import Dict, Tuple


def kNN(train: torch.Tensor,
        test: torch.Tensor,
        k: int) -> torch.Tensor:
    """
    for a given test batch, finds the pairwise knn for every sample in the given train batch
    :param train: a batch from the train set
    :param test: a batch from the test set
    :param k: neighbours to consider (hyper parameter)
    :return: pairwise k-nearest-neighbours
    """
    dist = torch.cdist(test.flatten(1), train.flatten(1))
    knn = dist.topk(k, largest=False)
    return knn


def _calc_current_knn_and_append(curr_train: torch.Tensor,
                                 curr_test: torch.Tensor,
                                 dict_to_update: Dict[str, torch.Tensor],
                                 k: int) -> None:
    """
    for a given train batch and test batch (both of specific type - deep/shallow), calculated the pairwise kNN
    and appends both values and indices to a provided return tensor
    """
    curr_knn = kNN(curr_train, curr_test, k)
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
    shallow_ret, deep_ret = torch.tensor([]), torch.tensor([])
    for curr_test_shallow, curr_test_deep in tqdm(test):  # TODO: maybe iterate both reg & anomal using zip(..)

        knn_shallow = {'values': torch.tensor([]), 'indices': torch.tensor([])}
        knn_deep = {'values': torch.tensor([]), 'indices': torch.tensor([])}

        for curr_train_shallow, curr_train_deep in tqdm(train):
            _calc_current_knn_and_append(curr_train_shallow, curr_test_shallow, knn_shallow, k)
            _calc_current_knn_and_append(curr_train_deep, curr_test_deep, knn_deep, k)

        shallow_ret = torch.cat((shallow_ret, _get_predictions(knn_shallow, k)))
        deep_ret = torch.cat((deep_ret, _get_predictions(knn_deep, k)))
    torch.save(deep_ret, f'predictions/deep_{dataset}')
    torch.save(shallow_ret, f'predictions/shallow_{dataset}')
    return shallow_ret, deep_ret


def load_predictions_and_measure_accuracy() -> Dict[str, Dict[str, torch.Tensor]]:
    """
    loads the already calculated predictions and calculates relevant measurements
    :return: a nested dict of the form {ft/der/shallow/deep: {prediction: ,classification: ,accuracy:},... }
    """
    data = {}
    deep_cifar10 = torch.load('predictions/deep_cifar10')
    shallow_cifar10 = torch.load('predictions/shallow_cifar10')
    shallow_mnist = torch.load('predictions/shallow_mnist')
    deep_mnist = torch.load('predictions/deep_mnist')
    x = 2
