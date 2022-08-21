from Data_Preprocess import *
from DCkNN import *
from matplotlib import pyplot as plt
from tabulate import tabulate
from sklearn.metrics import roc_curve, roc_auc_score
from Utils import score_prediction
import numpy as np


def find_hyperparams_and_plot_ROC(anomal_class, reg_class):
    deep_anomal, deep_reg, mid_anomal, mid_reg, shal_anomal, shal_reg = load_activations(anomal_class, reg_class)

    size = 600
    deep_range = [0.3 + diff / 100 for diff in range(size)]
    mid_range = [140 + diff / 10 for diff in range(size)]
    shal_range = [1500 + diff for diff in range(size)]
    thresholds = [{'deep': d, 'mid': m, 'shallow': s} for d, m, s
                  in zip(deep_range, mid_range, shal_range)]
    ROC_x = []
    ROC_y = []
    for t in thresholds:
        # print(t)
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

    AUC = sum([ROC_y[i] * 1 / size for i in range(size)])

    plt.scatter(ROC_x, ROC_y, s=10)
    plt.xlabel("FPR")
    plt.title(f"ROC Curve, AUC={AUC:.5f}")
    plt.grid()
    plt.show()


def load_activation_dataloaders(calc_activations: bool,
                                calc_anomal_activations: bool,
                                anomalous_class: str,
                                regular_class: str,
                                train_size: int,
                                test_size: int,
                                one_vs_other: bool,
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
                                                                             regular_class, one_vs_other, args)

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


def load_activations(anomal_class, reg_class):
    deep_reg = torch.load(f'predictions/deep_{reg_class}_regular')
    mid_reg = torch.load(f'predictions/mid_{reg_class}_regular')
    shal_reg = torch.load(f'predictions/shallow_{reg_class}_regular')
    deep_anomal = torch.load(f'predictions/deep_{anomal_class}_anomal')
    mid_anomal = torch.load(f'predictions/mid_{anomal_class}_anomal')
    shal_anomal = torch.load(f'predictions/shallow_{anomal_class}_anomal')
    return deep_anomal, deep_reg, mid_anomal, mid_reg, shal_anomal, shal_reg


def get_roc_curve(anomal_class: str, reg_class: str):
    deep_anomal, deep_reg, mid_anomal, mid_reg, shallow_anomal, shallow_reg = load_activations(anomal_class, reg_class)
    anomal_size, regular_size = deep_anomal.shape, deep_reg.shape
    # 1 for anomal, 0 for regular
    y_true = torch.cat((torch.ones(anomal_size), torch.zeros(regular_size)))

    deep_knn_dist = torch.cat((deep_anomal, deep_reg))
    mid_knn_dist = torch.cat((mid_anomal, mid_reg))
    shallow_knn_dist = torch.cat((shallow_anomal, shallow_reg))

    # sklearn roc_curve can take the distances themselves (and not the predictions)
    deep_fpr, deep_tpr, deep_thresholds = roc_curve(y_true, deep_knn_dist)
    mid_fpr, mid_tpr, mid_thresholds = roc_curve(y_true, mid_knn_dist)
    shallow_fpr, shallow_tpr, shallow_thresholds = roc_curve(y_true, shallow_knn_dist)

    # Now, we know that the best point on the TPR-vs-FPR curve is (x,y) = (FPR,TPR) = (0,1)
    # we will chose the threshold that corresponds to the point that is closest to (0,1), i.e the point (x,y) for which
    # sqrt(x^2+(y-1)^2) is minimal, and use this to extract the threshold (we do not take sqrt as this is redundant):
    best_deep_threshold = deep_thresholds[np.argmin(deep_fpr ** 2 + (1 - deep_tpr) ** 2)]
    best_mid_threshold = mid_thresholds[np.argmin(mid_fpr ** 2 + (1 - mid_tpr) ** 2)]
    best_shallow_threshold = shallow_thresholds[np.argmin(shallow_fpr ** 2 + (1 - shallow_tpr) ** 2)]

    # based on the threshold we generate a prediction - 1 = anomal, 0 = regular
    deep_pred = torch.where(deep_knn_dist <= best_deep_threshold, 0, 1)
    mid_pred = torch.where(mid_knn_dist <= best_mid_threshold, 0, 1)
    shallow_pred = torch.where(shallow_knn_dist <= best_shallow_threshold, 0, 1)

    # our majority vote function takes a dictionary, so we create it here
    prediction_dict = {'deep': deep_pred, 'mid': mid_pred, 'shallow': shallow_pred}

    final_prediction = majority_vote(prediction_dict)
    fpr, tpr, thresholds = roc_curve(y_true, final_prediction)

    deep_fpr, deep_tpr, deep_thresholds = roc_curve(y_true, deep_pred)
    mid_fpr, mid_tpr, mid_thresholds = roc_curve(y_true, mid_pred)
    shallow_fpr, shallow_tpr, shallow_thresholds = roc_curve(y_true, shallow_pred)

    # x = fpr, y = tpr
    plt.plot(deep_fpr, deep_tpr)
    plt.plot(mid_fpr, mid_tpr)
    plt.plot(shallow_fpr, shallow_tpr)
    plt.plot(fpr, tpr)
    plt.legend([f"Deep {roc_auc_score(y_true, deep_pred):.4f}",
                f"Mid {roc_auc_score(y_true, mid_pred):.4f}",
                f"Shallow {roc_auc_score(y_true, shallow_pred):.4f}",
                f"Final {roc_auc_score(y_true, final_prediction):.4f}"])
    plt.show()
    x = 2
    # TODO: try using the knn distances themselves as maj vote and not the final pred values


def main(args):
    k = 2
    train_size = 1000
    test_size = 1000
    regular_class = 'mnistcls'
    anomalous_class = 'mnist'

    calc_reg_activation = True
    calc_anomal_activation = True

    use_one_vs_other = True
    args = args if use_one_vs_other else None
    # if use_one_vs_other: anomalous_class = 'cifar10'

    calculate_knn = True
    calculate_knn_anomalous = True

    visualize = True

    do_ROC = True

    # TODO: remove from here once done testing
    get_roc_curve(anomalous_class, regular_class)

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
                    ['Train Size', train_size],
                    ['Test size', test_size]],
                   tablefmt="fancy_grid"))

    reg_train, reg_test, anomal_test = load_activation_dataloaders(calc_reg_activation, calc_anomal_activation,
                                                                   anomalous_class, regular_class, train_size,
                                                                   test_size, use_one_vs_other, args)

    if calculate_knn: committee_kNN_from_all_files(k, reg_train, reg_test, regular_class, False)
    if calculate_knn_anomalous: committee_kNN_from_all_files(k, reg_train, anomal_test, anomalous_class, True)

    if visualize: visualize_results(anomalous_class, regular_class, k, test_size)

    if do_ROC: find_hyperparams_and_plot_ROC(anomalous_class, regular_class)


if __name__ == '__main__':
    main(0)
