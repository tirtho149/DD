# Drowsiness Detection Project

This project aims to detect drowsiness using a combination of computer vision and speech recognition. It includes scripts for training a drowsiness detection model and a dynamic webpage to control the training process.

## Overview

Drowsiness detection is a crucial component in various applications, such as driver monitoring systems and workplace safety. This project employs computer vision techniques to analyze facial features and determine whether a person is exhibiting signs of drowsiness. Additionally, it incorporates speech recognition for hands-free control of the training process.

## Project Components

### 1. Model Training

- `train_drowsiness_model.py`: This Python script utilizes TensorFlow and Keras to train a convolutional neural network (CNN) for drowsiness detection. The model is trained on a dataset of facial images, distinguishing between open and closed eyes.

#### Usage:

```bash
python train_drowsiness_model.py --train_dir <path/to/train_data> --valid_dir <path/to/valid_data>
