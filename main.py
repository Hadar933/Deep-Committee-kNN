from Data_Preprocess import *
from DCkNN import *
from matplotlib import pyplot as plt
from tabulate import tabulate

from Utils import score_prediction


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
        calculate_activations_and_save(reg_train_loader, 'train', regular_class)
        calculate_activations_and_save(reg_test_loader, 'test', regular_class)

    if calc_anomal_activations:  # the anomalous is only used for testing
        calculate_activations_and_save(anomal_test_loader, 'test', anomalous_class)

    reg_train_activations_loader = ActivationDataset(regular_class, 'train', False)
    reg_test_activations_loader = ActivationDataset(regular_class, 'test', False)
    anomal_test_activations_loader = ActivationDataset(anomalous_class, 'test', True)

    return reg_train_activations_loader, reg_test_activations_loader, anomal_test_activations_loader


def visualize_results(anomal_class, reg_class, k):
    deep_reg = torch.load(f'predictions/deep_{reg_class}_regular')
    mid_reg = torch.load(f'predictions/mid_{reg_class}_regular')
    shal_reg = torch.load(f'predictions/shallow_{reg_class}_regular')

    deep_anomal = torch.load(f'predictions/deep_{anomal_class}_anomal')
    mid_anomal = torch.load(f'predictions/mid_{anomal_class}_anomal')
    shal_anomal = torch.load(f'predictions/shallow_{anomal_class}_anomal')

    x = range(1000)  # test size

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


def main():
    k = 2
    train_size = 1000
    test_size = 1000
    regular_class = 'cifar10'
    anomalous_class = 'mnist'

    calc_reg_activation = True
    calc_anomal_activation = True

    use_one_vs_other = False
    args = 0 if use_one_vs_other else None
    if use_one_vs_other: anomalous_class = f'cifar10cls'

    calculate_knn = True
    calculate_knn_anomalous = True

    visualize = True

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
                    ['Train Size', train_size],
                    ['Test size', test_size]],
                   tablefmt="fancy_grid"))

    cifar10_train, cifar10_test, mnist_test = load_activation_dataloaders(calc_reg_activation, calc_anomal_activation,
                                                                          anomalous_class, regular_class, train_size,
                                                                          test_size, use_one_vs_other, args)

    if calculate_knn: committee_kNN_from_all_files(k, cifar10_train, cifar10_test, regular_class, False)
    if calculate_knn_anomalous: committee_kNN_from_all_files(k, cifar10_train, mnist_test, anomalous_class, True)

    if visualize: visualize_results(anomalous_class, regular_class, k)

    if do_ROC: find_hyperparams_and_plot_ROC()


if __name__ == '__main__':
    main()
