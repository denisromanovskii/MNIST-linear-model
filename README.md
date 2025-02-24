<p align="center">
      <img src="https://i.ibb.co/mF2LmMhc/MNIST-linear-model.png](https://i.ibb.co/mF2LmMhc/MNIST-linear-model.png" width="726">
</p>

<p align="center">
   <img src="https://img.shields.io/badge/Language-Python_3.12-blue" alt="Python Version">
   <img src="https://img.shields.io/badge/Library-PyTorch_2.6.0-orange" alt="PyTorch Version">
   <img src="https://img.shields.io/badge/GUI-PyQt6-red" alt="PyQt Version">
   <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

## About

This project implements a linear neural network to recognize digits from the MNIST dataset. MNIST (Modified National Institute of Standards and Technology) is a widely used dataset for training and testing machine learning algorithms, consisting of 28x28 pixel images of handwritten digits from 0 to 9.
The project utilizes a simple linear architecture where data is passed through a single layer of neurons to demonstrate the basic principles of neural networks and their application to the task of image classification. 
The goal of the project is to demonstrate the basic principle of neural networks, and to show how these methods can be applied to classification problems.

## Demo

<p align="center">
      <img src="https://github.com/denisromanovskii/MNIST-linear-model/blob/main/MNIST_demo.gif" alt="demo gif" height="300px">
</p>

## Documentation

### Included Files:

- **-**  **`MNIST-model-params.pt`** - A file where the parameters (weights) of the trained neural network model are saved in PyTorch format.
- **-** **`MNISTdataset.py`** - A module that loads and processes the MNIST dataset, preparing it for training the model. It includes code for reading and transforming images and labels.
- **-** **`MNISTmodel.py`** - A module that defines the architecture of the neural network model for MNIST digit recognition. It includes the description of the layers and structure of the model.
- **-** **`dataPreparation.py`** - A module that handles the data preparation process for training, including normalization, splitting the dataset into training and testing sets, and any other necessary transformations.
- **-** **`main.py`** - A module that creates a user interface (UI) and allows users to draw digits. The model then guesses which digit the user has drawn.
- **-** **`test_model.py`** - A module for testing the trained model. It evaluates the model’s performance on the test dataset (e.g., calculating accuracy).
- **-** **`train_MNIST_model.py`** - A module for training the model on the MNIST dataset. It includes model setup, loss function, optimizer, and the process of training the model on the training data.
  

## Developers

- [Denis Romanovskii](https://github.com/denisromanovskii)

## License
Project MNIST-linear-model is destributed under the MIT license.
