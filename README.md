# SMML-
Urban Sound Classification with Neural Networks


the link for the Datset is https://urbansounddataset.weebly.com/urbansound8k.html

FeatureExtraction.py: file loading, data pre-processing and feature extraction with splitting to the right folders

MLArchitecture.py: feature loading and normalization. You can plot the features into a 2D space and choose what features to use for the architecture. Three models can be trained and tested, creating three: the confusion matrix, the accuracy and the loss function.

Folders:
- Images
	- RF: confusion matrix of all the features with all different numbers of estimators
	- Dense: confusion matrix of all the features trained with the ANN, with both batch_size = 64 and =128, a plot with loss and accuracy related to that feature and batch size and a plot with just the accuracy.
	- CNN: confusion matrix of all the features trained with the CNN, with both batch_size = 64 and =128, a plot with loss and accuracy related to that feature and batch size and a plot with just the accuracy.
	- VGG: confusion matrix of all the features trained using a customized version of the VGG, with both batch_size = 64 and =128, a plot with loss and accuracy related to that feature and batch size and a plot with just the accuracy.

- Train: it contains extracted training labels and only rmsTraining features because the others file are too large to be uploaded. If you need them, please contact me
- Test: it contains all the extracted features and the labels related to the test set split into 5 different folders.
