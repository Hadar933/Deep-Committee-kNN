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
                                 ):
    """
    :return: classifications of the test data
    """
    names_and_knns_dict = {
        'layer1_block0': torch.tensor([], device=device),
        'layer1_block1': torch.tensor([], device=device),

        'layer2_block0': torch.tensor([], device=device),
        'layer2_block1': torch.tensor([], device=device),

        'layer3_block0': torch.tensor([], device=device),
        'layer3_block1': torch.tensor([], device=device),

        'layer4_block0': torch.tensor([], device=device),
        'layer4_block1': torch.tensor([], device=device),

        'avgpool': torch.tensor([], device=device)
    }

    for curr_test_activ_lst in tqdm(test):
        curr_test_activ_lst = [curr_test_activ_lst[i].to(device) for i in range(len(curr_test_activ_lst))]

        # TODO: might not need indices here
        knn_values = {
            'layer1_block0': {'values': torch.tensor([], device=device), 'indices': torch.tensor([], device=device)},
            'layer1_block1': {'values': torch.tensor([], device=device), 'indices': torch.tensor([], device=device)},

            'layer2_block0': {'values': torch.tensor([], device=device), 'indices': torch.tensor([], device=device)},
            'layer2_block1': {'values': torch.tensor([], device=device), 'indices': torch.tensor([], device=device)},

            'layer3_block0': {'values': torch.tensor([], device=device), 'indices': torch.tensor([], device=device)},
            'layer3_block1': {'values': torch.tensor([], device=device), 'indices': torch.tensor([], device=device)},

            'layer4_block0': {'values': torch.tensor([], device=device), 'indices': torch.tensor([], device=device)},
            'layer4_block1': {'values': torch.tensor([], device=device), 'indices': torch.tensor([], device=device)},

            'avgpool': {'values': torch.tensor([], device=device), 'indices': torch.tensor([], device=device)},
        }

        for curr_train_activ_lst in tqdm(train):
            curr_train_activ_lst = [curr_train_activ_lst[i].to(device) for i in range(len(curr_train_activ_lst))]
            count_last_activ = 0
            for train_activ, test_activ, knn_dict in zip(curr_train_activ_lst, curr_test_activ_lst,
                                                         knn_values.values()):
                count_last_activ += 1
                is_deep = True if count_last_activ < 8 else False  # only last layer is deep and we have 9 layers
                _calc_current_knn_and_append(train_activ, test_activ, knn_dict, k, is_deep)
            count_last_activ = 0

        for name, knn in names_and_knns_dict.items():
            names_and_knns_dict[name] = torch.cat((names_and_knns_dict[name], _get_predictions(knn_values[name], k)))

    anomal_suffix = "anomal" if is_anomalous else "regular"
    for name, knn in names_and_knns_dict.items():
        torch.save(knn, f"predictions/{name}_{dataset}_{anomal_suffix}")

    return names_and_knns_dict


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
