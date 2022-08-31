# Deep Committee kNN
 
DCkNN is a general pipeline for anomaly detection im images. It used multiple activation maps that are given from a pre=trained ImageNet based neural network like ResNet, finds the k nearest neighbours of some test sample from every set of activations and uses a majority vote on all those to classify the sample as "normal" or "anomalous" 
