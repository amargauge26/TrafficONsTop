# TrafficONsTop

# Traffic Light Detection and Classification 🚦

This project utilizes pre-trained deep learning models for detecting and classifying traffic lights in images. The system combines object detection and image classification techniques to identify traffic lights and their states (Red, Green, Yellow). It can be used for autonomous driving systems or traffic monitoring applications.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation and Testing](#evaluation-and-testing)
- [Contributing](#contributing)
- [License](#license)

## Overview 📸

This project performs two main tasks:
1. **Traffic Light Detection**: Detect traffic lights in images using an SSD MobileNet pre-trained object detection model.
2. **Traffic Light Classification**: Classify the detected traffic light's state (Red, Green, Yellow) using transfer learning with an InceptionV3 model.

### Key Components
- **TLClassifier Class**: Loads the object detection model and performs traffic light detection on input images.
- **InceptionV3 Model**: Fine-tuned for classifying the state of the traffic light (Red, Green, Yellow).
- **Data Augmentation**: Used to increase dataset diversity and improve model generalization.

## Features 🌟
- **Traffic Light Detection**: Identifies traffic lights in images using SSD MobileNet.
- **Traffic Light State Classification**: Classifies the state of traffic lights (Red, Green, Yellow) using transfer learning.
- **Data Augmentation**: Enhances model robustness by using techniques like flipping, rotation, and zoom.
- **Performance Metrics**: Evaluates the model using accuracy, precision, recall, F1-score, and confusion matrix.

## Technologies Used ⚙️
- **TensorFlow/Keras**: For model training and evaluation.
- **OpenCV**: For image processing and visualization.
- **NumPy/Pandas**: For data manipulation.
- **Matplotlib**: For visualizing performance metrics and confusion matrix.
- **SSD MobileNet**: Pre-trained object detection model for traffic light detection.
- **InceptionV3**: Pre-trained classification model for traffic light state recognition.

## Setup Instructions 🛠️

### Prerequisites
1. Python 3.7+
2. Install dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
