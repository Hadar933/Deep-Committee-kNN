from Data_Preprocess import *
from DCkNN import *


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


# TODO: probably need to normalize the activations.
def main():
    k = 3
    train_size = 5000
    regular_class = 'cifar10'
    anomalous_class = 'mnist'
    calc_reg_activation = True
    calc_anomal_activation = True
    calculate_knn = True
    calculate_knn_anomalous = True

    cifar10_train, cifar10_test, mnist_test = load_activation_dataloaders(calc_reg_activation, calc_anomal_activation,
                                                                          anomalous_class, regular_class, train_size)
    if calculate_knn: committee_kNN_from_all_files(k, cifar10_train, cifar10_test, regular_class)
    if calculate_knn_anomalous: committee_kNN_from_all_files(k, cifar10_train, mnist_test, anomalous_class)

    load_predictions_and_measure_accuracy()


if __name__ == '__main__':
    ret = main()
