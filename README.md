# Conser-vision Wildlife Image Classification - WorldQuant University Project

[![DrivenData](https://img.shields.io/badge/DrivenData-Project-blue)](https://www.drivendata.org/)
[![WorldQuant University](https://img.shields.io/badge/WorldQuant_University-Project-041E42)](https://wqu.org/)
[![Image Classification](https://img.shields.io/badge/Task-Image_Classification-brightgreen)](https://en.wikipedia.org/wiki/Image_classification)
[![Python](https://img.shields.io/badge/Python-3.7+-yellow.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-orange.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

## Overview

This repository contains the code and documentation for the Conser-vision Wildlife Image Classification project, completed as part of a WorldQuant University course. The project addresses a data science competition on DrivenData, aiming to classify wildlife images from camera traps using computer vision and deep learning techniques.

**This project directly addresses the following learning objectives from the WorldQuant University course:**

*   **Reading and Preparing Image Files:**  Demonstrates how to load and preprocess image data for machine learning models.
*   **PyTorch Implementation:**  Utilizes PyTorch to build and manipulate tensors and construct neural network models.
*   **Convolutional Neural Networks (CNNs):**  Focuses on building effective CNN architectures specifically designed for image classification tasks.
*   **Prediction on New Images:**  Shows how to use the trained model to make predictions on unseen image data.
*   **Competition Submission:**  Includes code to generate submission files in the format required for the DrivenData competition.

## Business Problem (from WorldQuant University Project Description)

> In this project, you'll look at a data science competition helping scientists track animals in a wildlife preserve. The goal is to take images from camera traps and classify which animal, if any, is present. To complete the competition, you'll expand your machine learning skills by creating more powerful neural network models that can take images as inputs and classify them into one of multiple categories.

## Project Structure

Conser-vision-Image-Classification-WQU/
├── data/
│   └── [Instructions on how to download data - README.md]
├── notebooks/
│   └── 1_data_loading_and_preparation.ipynb
│   └── 2_cnn_model_building_pytorch.ipynb
│   └── 3_model_training.ipynb
│   └── 4_prediction_and_submission.ipynb
├── documentation/
│   └── Project_Overview_WQU.md
└── README.md

*   **`data/`**: Contains instructions to download the dataset from DrivenData. **Raw data files are NOT included in the repository.**
*   **`notebooks/`**: Jupyter Notebooks, each focused on a specific learning objective.
*   **`documentation/`**: Project overview and learning objectives from WorldQuant University.
*   **`README.md`**: Project overview, learning objectives summary, and repository guide (this file).

## Notebooks - Addressing Learning Objectives

1.  **`notebooks/1_data_loading_and_preparation.ipynb`**:
    *   **Learning Objective Addressed:** How to read image files and prepare them for machine learning.
    *   **Content:** Code for loading image data from the DrivenData dataset, preprocessing steps (e.g., resizing, normalization), and creating data loaders for PyTorch.

2.  **`notebooks/2_cnn_model_building_pytorch.ipynb`**:
    *   **Learning Objectives Addressed:** How to use PyTorch to manipulate tensors and build a neural network model, How to build a Convolutional Neural Network that works well with images.
    *   **Content:** Implementation of a Convolutional Neural Network (CNN) model in PyTorch for image classification.  Demonstrates tensor manipulation, layer definition, and CNN architecture design.

3.  **`notebooks/3_model_training.ipynb`**:
    *   **Content:** Code for training the CNN model using the prepared data loaders and PyTorch training loops. Includes model evaluation on validation data and performance metrics.

4.  **`notebooks/4_prediction_and_submission.ipynb`**:
    *   **Learning Objectives Addressed:** How to use that model to make predictions on new images, How to turn those predictions into a submission to the competition.
    *   **Content:**  Code for loading the trained model, making predictions on the test dataset, and generating the submission file in the required CSV format for DrivenData.

## Documentation

*   **[Project Overview & WQU Learning Objectives](documentation/Project_Overview_WQU.md)**: Contains the project description from WorldQuant University, explicitly listing the learning objectives and how this project fulfills them.

## Get Started

1.  **Clone the repository:** `git clone [repository URL]`
2.  **Download the Dataset:** Follow the instructions in `data/README.md` to download the dataset from DrivenData.
3.  **Explore the Notebooks:**  Run the Jupyter Notebooks in the `notebooks/` folder sequentially to follow the project from data loading to submission.
