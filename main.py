import torch

from Data_Preprocess import *
from DCkNN import *
from matplotlib import pyplot as plt


def load_activation_dataloaders(calc_activations: bool,
                                calc_anomal_activations: bool,
                                anomalous_class: str,
                                regular_class: str,
                                train_size: int
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
    cifar10_train_loader, cifar10_test_loader, mnist_test_loader = load_dataloaders(train_size)

    if calc_activations:
        calculate_activations_and_save(cifar10_train_loader, 'train', regular_class)
        calculate_activations_and_save(cifar10_test_loader, 'test', regular_class)
    if calc_anomal_activations:  # the anomalous is only used for testing
        calculate_activations_and_save(mnist_test_loader, 'test', anomalous_class)

    cifar10_train_activations_loader = ActivationDataset(regular_class, 'train')
    cifar10_test_activations_loader = ActivationDataset(regular_class, 'test')
    mnist_test_activations_loader = ActivationDataset(anomalous_class, 'test')

    return cifar10_train_activations_loader, cifar10_test_activations_loader, mnist_test_activations_loader


def visualize_results():
    deep_cifar10, shallow_cifar10, shallow_mnist, deep_mnist = load_predictions()
    x = range(10000)

    plt.scatter(x, shallow_cifar10.cpu().numpy(), s=0.3)
    plt.scatter(x, shallow_mnist.cpu().numpy(), s=0.3)
    plt.title("Shallow Activations")
    plt.xlabel("# Sample")
    plt.ylabel("Mean kNN distance from k=3 neighbours")
    plt.legend(['CIFAR10', 'MNIST'])
    plt.show()

    plt.scatter(x, deep_cifar10.cpu().numpy(), s=0.3)
    plt.scatter(x, deep_mnist.cpu().numpy(), s=0.3)
    plt.title("Deep Activations")
    plt.xlabel("# Sample")
    plt.ylabel(f"Mean kNN distance from k=3 neighbours")
    plt.legend(['CIFAR10', 'MNIST'])
    plt.show()


def main():
    k = 2
    train_size = 5000
    regular_class = 'cifar10'
    anomalous_class = 'mnist'
    calc_reg_activation = False
    calc_anomal_activation = False
    calculate_knn = True
    calculate_knn_anomalous = True

    cifar10_train, cifar10_test, mnist_test = load_activation_dataloaders(calc_reg_activation, calc_anomal_activation,
                                                                          anomalous_class, regular_class, train_size)
    if calculate_knn: committee_kNN_from_all_files(k, cifar10_train, cifar10_test, regular_class)
    if calculate_knn_anomalous: committee_kNN_from_all_files(k, cifar10_train, mnist_test, anomalous_class)


if __name__ == '__main__':
    main()