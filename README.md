# Deep Committee kNN
 
DCkNN is a general pipeline for anomaly detection in images. It used multiple activation maps that are given from a pre-trained ImageNet based neural network like ResNet, finds the k nearest neighbours of some test sample from every set of activations and uses a majority vote on all those to classify the sample as "normal" or "anomalous"

<p align="center">
<img src="https://user-images.githubusercontent.com/45313790/187689119-9737ba63-4e20-42eb-b463-bdc0b5967601.png" width=548 height=400>
</p>
