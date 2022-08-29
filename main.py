from Data_Preprocess import *
from DCkNN import *
from matplotlib import pyplot as plt
from tabulate import tabulate
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import numpy as np


def load_activation_dataloaders(calc_activations: bool,
                                calc_anomal_activations: bool,
                                anomalous_class: str,
                                regular_class: str,
                                train_size: int,
                                test_size: int,
                                one_vs_other: bool,
                                use_retinal_dataset: bool,
                                *args: int
                                ) -> Tuple[ActivationDataset, ActivationDataset, ActivationDataset]:
    """
    takes the original regular and anomalous dataloaders and converts them to dataloaders of activations
    :param calc_activations: True iff you wish to calculate the regular class activations
    :param calc_anomal_activations: True iff you wish to calculate the anomaly class activations
    :param anomalous_class: anomaly class name
    :param regular_class: regular class name
    :param train_size: the size we wish to use as training ( = subset of the original training size)
    :return: three activation dataloaders for each dataset
    """
    reg_train_loader, reg_test_loader, anomal_test_loader = load_dataloaders(train_size, test_size, anomalous_class,
                                                                             regular_class, one_vs_other,
                                                                             use_retinal_dataset, args)

    if calc_activations:
        calculate_activations_and_save(reg_train_loader, 'train', regular_class, False)
        calculate_activations_and_save(reg_test_loader, 'test', regular_class, False)

    if calc_anomal_activations:  # the anomalous is only used for testing
        calculate_activations_and_save(anomal_test_loader, 'test', anomalous_class, True)

    reg_train_activations_loader = ActivationDataset(regular_class, 'train', False)
    reg_test_activations_loader = ActivationDataset(regular_class, 'test', False)
    anomal_test_activations_loader = ActivationDataset(anomalous_class, 'test', True)

    return reg_train_activations_loader, reg_test_activations_loader, anomal_test_activations_loader


def visualize_results(anomal_class: str, reg_class: str, k: int, test_size):
    deep_anomal, deep_reg, mid_anomal, mid_reg, shal_anomal, shal_reg = load_activations(anomal_class, reg_class)

    x = range(test_size)

    for activ in ['S', 'M', 'D']:
        reg_plot = shal_reg if activ == 'S' else mid_reg if activ == 'M' else deep_reg
        anomal_plot = shal_anomal if activ == 'S' else mid_anomal if activ == 'M' else deep_anomal
        reg_plot = reg_plot.cpu().numpy()
        anomal_plot = anomal_plot.cpu().numpy()

        plt.scatter(x, reg_plot, s=0.3)
        plt.scatter(x, anomal_plot, s=0.3)
        plt.title(f"{activ} Activations")
        plt.xlabel("# Sample")
        plt.ylabel(f"Mean kNN distance from k={k} neighbours")
        plt.legend([reg_class, anomal_class])
        plt.show()


def load_activations(anomal_class, reg_class, to_cpu: bool):
    anomal_act, regular_act = {}, {}
    for act in [f"layer{j}_block{i}" for j in [1, 2, 3, 4] for i in [0, 1]] + ['avgpool']:
        for suffix in ["regular", "anomal"]:
            if suffix == "regular":
                val = torch.load(f"predictions/{act}_{reg_class}_{suffix}")
                if to_cpu:
                    regular_act[act] = val.cpu()
                else:
                    regular_act[act] = val

            elif suffix == "anomal":
                val = torch.load(f"predictions/{act}_{anomal_class}_{suffix}")
                if to_cpu:
                    anomal_act[act] = val.cpu()
                else:
                    anomal_act[act] = val

    return anomal_act, regular_act


def get_roc_curve(anomal_class: str, reg_class: str, use_knn_sum: bool):
    anomal_act, regular_act = load_activations(anomal_class, reg_class, True)

    anomal_size, regular_size = anomal_act['layer1_block0'].shape, regular_act['layer1_block0'].shape
    # 1 for anomal, 0 for regular
    y_true = torch.cat((torch.ones(anomal_size), torch.zeros(regular_size)))

    layers_knn_dict = {}
    for layer_name in [f"layer{j}_block{i}" for j in [1, 2, 3, 4] for i in [0, 1]] + ['avgpool']:
        knn_dist = torch.cat((anomal_act[layer_name], regular_act[layer_name]))
        layers_knn_dict[layer_name] = knn_dist
    if use_knn_sum:
        final_knn = 0
        for knn in layers_knn_dict.values():
            final_knn += knn
        final_fpr, final_tpr, final_thres = roc_curve(y_true, final_knn)
        best_threshold = final_thres[np.argmin(final_fpr ** 2 + (1 - final_tpr) ** 2)]
        y_pred = torch.where(final_knn <= best_threshold, 0, 1)
        # TODO: continue from here !!
    prediction_dict = {}
    tpr_dict = {}
    fpr_dict = {}
    for layer_name, knn in layers_knn_dict.items():
        # sklearn roc_curve can take the distances themselves (and not the predictions)
        fpr, tpr, thresholds = roc_curve(y_true, knn)

        # Now, we know that the best point on the TPR-vs-FPR curve is (x,y) = (FPR,TPR) = (0,1)
        # we will chose the threshold that corresponds to the point that is closest to (0,1), i.e the point (x,y) for
        # which sqrt(x^2+(y-1)^2) is minimal, and use this to extract the threshold (taking sqrt is redundant):
        best_threshold = thresholds[np.argmin(fpr ** 2 + (1 - tpr) ** 2)]

        # based on the threshold we generate a prediction - 1 = anomal, 0 = regular:
        prediction = torch.where(knn <= best_threshold, 0, 1)

        prediction_dict[layer_name] = prediction

        # we also re calculate the roc curve based on the best threshold for each layer
        fpr, tpr, thresholds = roc_curve(y_true, prediction)
        tpr_dict[layer_name] = tpr
        fpr_dict[layer_name] = fpr

    # finally we take a makority vote of all predictions
    final_prediction = majority_vote(prediction_dict)
    final_fpr, final_tpr, final_thresholds = roc_curve(y_true, final_prediction)

    # we now plot all our results where x axis = fpr, y axis = tpr
    for curr_fpr, curr_tpr in zip(fpr_dict.values(), tpr_dict.values()):
        plt.plot(curr_fpr, curr_tpr)
    plt.plot(final_fpr, final_tpr)
    plt.xlabel("FPR"), plt.ylabel("TPR"), plt.title("ROC Curve")
    plt.legend([f"{layer_name} ({roc_auc_score(y_true, curr_pred):.4f})" for layer_name, curr_pred in
                prediction_dict.items()] + [f'Final pred ({roc_auc_score(y_true, final_prediction):.4f})'])
    plt.show()

    for layer_name, pred in prediction_dict.items():
        print(layer_name)
        print(confusion_matrix(y_true, pred))
    print('Final')
    print(confusion_matrix(y_true, final_prediction))


def main(args):
    k = 2
    train_size = 1000
    test_size = 1000
    regular_class = 'mknet'
    anomalous_class = 'mknet'

    use_retinal_dataset = True

    calc_reg_activation = True
    calc_anomal_activation = True

    use_one_vs_other = False
    args = args if use_one_vs_other else None

    calculate_knn = True
    calculate_knn_anomalous = True

    visualize = False

    do_ROC = True

    print("        Running with the following setup:")
    print(tabulate([['Regular Data', regular_class],
                    ['Anomalous Data', anomalous_class],
                    ['One-vs-Other (for CIFAR10 only)', use_one_vs_other],
                    ['Calculating Regular Activations', calc_reg_activation],
                    ['Calculating Anomalous Activations', calc_anomal_activation],
                    ['Calculating Regular kNN', calculate_knn],
                    ['Calculating Anomalous kNN', calculate_knn_anomalous],
                    ['Visualizing Data', visualize],
                    ['Plotting ROC', do_ROC],
                    ['Number of neighbours (k)', k],
                    ['Regular class instance ', f"class #{args}"],
                    ['Train Size', train_size],
                    ['Test size', test_size]],
                   tablefmt="fancy_grid"))

    reg_train, reg_test, anomal_test = load_activation_dataloaders(calc_reg_activation, calc_anomal_activation,
                                                                   anomalous_class, regular_class, train_size,
                                                                   test_size, use_one_vs_other, use_retinal_dataset,
                                                                   args)

    if calculate_knn: committee_kNN_from_all_files(k, reg_train, reg_test, regular_class, False)
    if calculate_knn_anomalous: committee_kNN_from_all_files(k, reg_train, anomal_test, anomalous_class, True)

    if visualize: visualize_results(anomalous_class, regular_class, k, test_size)

    if do_ROC: get_roc_curve(anomalous_class, regular_class, True)


if __name__ == '__main__':
    main(0)
