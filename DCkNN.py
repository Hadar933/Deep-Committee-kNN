import os
import torch
from tqdm import tqdm
from typing import Dict, Tuple
from Utils import device, emd_3d, ANOMAL_DATASETS


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
    return final_topk.values.mean(axis=1)


def committee_kNN_from_all_files(k: int,
                                 train,
                                 test,
                                 dataset: str,
                                 is_anomalous: bool
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :return: classifications of the test data
    """
    shallow_ret, mid_ret, deep_ret = torch.tensor([], device=device), torch.tensor([], device=device), torch.tensor([],
                                                                                                                    device=device)
    for curr_test_shallow, curr_test_mid, curr_test_deep in tqdm(test):
        curr_test_shallow = curr_test_shallow.to(device)
        curr_test_mid = curr_test_mid.to(device)
        curr_test_deep = curr_test_deep.to(device)

        # TODO: might not need indices here
        knn_shallow = {'values': torch.tensor([], device=device), 'indices': torch.tensor([], device=device)}
        knn_mid = {'values': torch.tensor([], device=device), 'indices': torch.tensor([], device=device)}
        knn_deep = {'values': torch.tensor([], device=device), 'indices': torch.tensor([], device=device)}

        for curr_train_shallow, curr_train_mid, curr_train_deep in tqdm(train):
            curr_train_shallow = curr_train_shallow.to(device)
            curr_train_mid = curr_train_mid.to(device)
            curr_train_deep = curr_train_deep.to(device)

            _calc_current_knn_and_append(curr_train_shallow, curr_test_shallow, knn_shallow, k, True)
            _calc_current_knn_and_append(curr_train_mid, curr_test_mid, knn_mid, k, True)
            _calc_current_knn_and_append(curr_train_deep, curr_test_deep, knn_deep, k, False)

        shallow_ret = torch.cat((shallow_ret, _get_predictions(knn_shallow, k)))
        mid_ret = torch.cat((mid_ret, _get_predictions(knn_mid, k)))
        deep_ret = torch.cat((deep_ret, _get_predictions(knn_deep, k)))

    anomal_suffix = "anomal" if is_anomalous else "regular"
    torch.save(deep_ret, f'predictions/deep_{dataset}_{anomal_suffix}')
    torch.save(mid_ret, f'predictions/mid_{dataset}_{anomal_suffix}')
    torch.save(shallow_ret, f'predictions/shallow_{dataset}_{anomal_suffix}')
    return shallow_ret, mid_ret, deep_ret


def classify_based_on_knn_distance(thresholds) -> Tuple[Dict[str, torch.ByteTensor], Dict[str, torch.ByteTensor]]:
    """
    loads the already calculated knn distances and returns a classification tensor
    it does that for every one of the activation maps.
    :return: anomalous and regular classifications
    """
    # TODO: make sure this func still works
    anomal_classifications = {}
    reg_classifications = {}
    files = os.listdir('predictions')
    for filename in files:
        activation = torch.load(f"predictions/{filename}")
        layer, dataset, is_anomalous = filename.split("_")
        thres = thresholds[layer]
        pred = activation <= thres  # True iff NOT Anomalous
        if dataset in ANOMAL_DATASETS:
            anomal_classifications[filename] = pred
        elif dataset == 'cifar10':
            reg_classifications[filename] = pred
    return anomal_classifications, reg_classifications


def majority_vote(classifications: Dict[str, torch.ByteTensor]):
    """
    stacks the classifications from all activations and returns the majority vote
    """
    stacked_classifications = torch.vstack([item for item in classifications.values()])
    vote = torch.mode(stacked_classifications, 0)
    return vote.values
