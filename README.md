# Face expression recognition with Deep Learning
This repository contains a Convolutional Neural Network (CNN) model for facial expression recognition. The project aims to classify images of faces into one of seven emotion categories using the FER-2013 dataset. The model is built and trained using Keras and TensorFlow.
 
## Table of Contents
1. Introduction
2. Dataset
3. Model Architecture
4. Training Process
5. Results and Visualizations
6. Limitations and Challenges
7. Future Work



### Introduction
Facial expression recognition is a challenging problem with numerous applications in human-computer interaction, security, and entertainment. This project leverages a deep learning approach using CNNs to classify facial expressions into seven categories: Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral.

## Dataset
The FER-2013 dataset consists of 48x48 pixel grayscale images of faces. The dataset is divided into training, validation, and test sets. The dataset is highly imbalanced and contains the following distribution of emotions:

- **Training Images:** 80% (28,800)
- **Validation Images:** 20% (7,200)
- **Image Size:** 48x48 pixels


## Model Architecture
- The CNN model is built using Keras with the following architecture:

1. Input Layer: 48x48 grayscale images
2. Convolutional Layers: 3 layers with 32, 64, and 128 filters respectively, each followed by ReLU activation and max-pooling
3. Dropout Layers: Applied after each max-pooling layer to prevent overfitting
4. Fully Connected Layers: Two layers with 256 and 128 units, each followed by ReLU activation
5. Output Layer: Softmax activation with 7 units (one for each emotion category)


## Training Process
- The model is trained using the Adam optimizer with a learning rate of 0.001. The training process includes:

* Batch Size: 64
* Epochs: 50
* Loss Function: Categorical Crossentropy
* The training history is saved for analysis and visualization.


## Results and Visualizations
- Confusion Matrix : The model achieved a validation accuracy of around 60-65% after 50 epochs.


## Limitations and Challenges
1. Imbalanced Dataset: The dataset is highly imbalanced, which affects the model's ability to generalize well across all classes.
2. Overfitting: The model tends to overfit after 20 epochs, as evidenced by the increasing gap between training and validation loss.
3. Low-Resolution Images: The images are low-resolution (48x48 pixels), which may limit the model's ability to capture fine details of facial expressions.

## Solutions Implemented
1. Dropout Layers: Added dropout layers to reduce overfitting.
2. Early Stopping: Considered early stopping to prevent overfitting during training.
3. Data Augmentation: Experimented with data augmentation to increase dataset diversity.

## Future Work
1. Improve Model Architecture: Experiment with more advanced architectures like ResNet or Inception.
2. Hyperparameter Tuning: Perform extensive hyperparameter tuning to find the optimal configuration.
3. Transfer Learning: Leverage pre-trained models on larger datasets for better feature extraction.




















