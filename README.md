# ResNet18 Hand Sign Classification for Auslan Dataset

This repository contains the implementation of a **ResNet18** model to classify hand signs from the **Auslan (Australian Sign Language)** dataset. The project uses **PyTorch** for building and training the model, and **Weights & Biases (W&B)** for tracking experiments, logging metrics, and visualizing results.

---

## Project Overview

This project uses the **ResNet18** architecture to classify grayscale hand signs from the Auslan dataset. The dataset contains 36 classes, representing the letters **A-Z** and the digits **0-9**.

### Key Features:
- **Grayscale Image Support**: The ResNet18 model is adapted to handle grayscale (1-channel) images.
- **Customizable**: Easily configurable hyperparameters such as learning rate, batch size, and number of epochs.
- **W&B Logging**: Tracks performance metrics (accuracy, precision, recall, F1-score) and logs confusion matrix and model artifacts.
  
---

## Jupyter Notebooks

### 1. Data Preparation

- **.py Code File**: `split_dataset.py`
- **Description**: 
  - This code file handles the preparation of the dataset by splitting it into training, validation, and testing sets.
  - Output: Split dataset ready for training.

---

### 2. Training

- **Notebook**: `training_model.ipynb`
- **Description**:
  - This notebook defines and trains the **ResNet18** model on the preprocessed Auslan dataset.
  - Logs metrics (e.g., accuracy, F1-score) and visualizations (e.g., confusion matrix) to **Weights & Biases (W&B)**.
  - Includes options for early stopping and model checkpointing.

- **Key Sections**:
  - **Model Definition**: Modifies the ResNet18 model to handle grayscale images and 36 classes.
  - **Training Loop**: Implements a custom training loop with metric logging and progress visualization using `tqdm`.
  - **W&B Integration**: Logs metrics to W&B for tracking experiments.

---

### 3. Testing

- **Notebook**: `testing_model.ipynb`
- **Description**:
  - This notebook tests the trained ResNet18 model on unseen data.
  - Outputs the classification accuracy, precision, recall, F1-score, and confusion matrix, logged to **W&B**.

- **Key Sections**:
  - **Test Loop**: Iterates through the test dataset, evaluates the model, and compares predictions with actual labels.
  - **W&B Logging**: Logs the final test results (confusion matrix, classification report) to W&B.

---

## Using Weights & Biases (W&B)

**W&B Integration** is a key part of the project for tracking and visualizing training and testing metrics.

### Key Metrics Logged:
- **Accuracy**: Overall classification accuracy during training and testing.
- **Precision, Recall, F1-Score**: Logged during both training and testing.
- **Confusion Matrix**: Shows misclassifications and performance across all classes.

### Instructions for Using W&B:
1. Install W&B:
   ```bash
   pip install wandb

## Results and Reports
### W&B reports for testing and training are included in this repository in .pdf format:
- ResNet18_handsign_train_W&B.pdf
- ResNet18_handsign_test_W&B.pdf
  
After completing training and testing, detailed results, including metrics and visualizations, can be viewed on Weights & Biases (W&B). To view the W&B report:

-  W&B Report Link is https://wandb.ai/srk_2024-the-australian-national-university/auslan-handsign-classification?nw=nwusersrk_2024
  
### The report includes:

- Confusion Matrix
- Precision, Recall, F1-score
- Loss and Accuracy curves

  ---
