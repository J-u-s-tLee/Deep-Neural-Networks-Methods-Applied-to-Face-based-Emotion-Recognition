# Deep Neural Networks Methods Applied to Face-based Emotion Recognition

---

## Overview
This repository contains a project exploring the use of a landmarked representation of image samples for facial emotion recognition. The primary objective is to evaluate whether a simpler representation of facial expressions using fiducial points (facial landmarks) is sufficient for emotion classification. Additionally, it investigates the impact of individual facial features, specifically the eyes and mouth, on the performance of a deep learning model.

---

## Methodology

### Dataset
*   The project utilizes the FER+ dataset, which features images labeled via crowd-sourcing by ten different annotators.
*   A majority voting strategy was applied to select a single emotion label for each image to be fed into the model.
*   The dataset images were divided into training (28,558 samples), validation (3,579 samples), and test (3,573 samples) sets.

### Feature Extraction
*   Google's MediaPipe Solutions, specifically the BlazeFace detector and face mesh model, were used to obtain facial landmarks.
*   The study compared four different input representations: original images, all 468 3-dimensional facial landmarks, landmarks of primary facial features (eyes, eyebrows, nose, and mouth), only eye landmarks, and only mouth landmarks.

### Model Architecture
*   A ResNet-18 (Residual Neural Network with 18 layers) model pretrained on the ImageNet database was utilized.
*   Images were resized to 224x224x3 and normalized prior to training.
*   The model's fully connected layer was modified to output probabilities for 9 specific classes and was optimized using Stochastic Gradient Descent (SGD) alongside a Cross-Entropy Loss function.

---

## Key Results
*   Using original image samples yielded the best overall performance, achieving a test accuracy of 79.21% and a balanced test accuracy of 69.8%.
*   There was a clear drop in the model's performance when using landmarked images compared to the original image samples.
*   Representing facial expressions using only the landmarks of major facial features (eyes, eyebrows, nose, and mouth) achieved a test accuracy of 57.10%, which was slightly higher than using all 468 facial landmarks (55.82% accuracy).
*   Models trained on mouth landmarks (50.67% accuracy) generally outperformed those trained solely on eye landmarks (42.17% accuracy), though eyes were better at representing the "Happiness" class.
*   The findings conclude that the relationship and combination of all facial features are highly important for accurate facial expression recognition.

---

## Academic Context
*   **Degree:** Licenciatura em Bioengenharia.
*   **Institution:** Faculdade de Engenharia da Universidade do Porto (FEUP).
*   **Author:** Leandro Miguel Pereira Ribeiro.
*   **Date:** June 18, 2024.
