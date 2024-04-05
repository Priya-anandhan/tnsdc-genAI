# Image generation using DCGAN

This project aims to develop a model using deep learning techniques, specifically DCGAN (Deep Convolutional Generative Adversarial Network), to generate realistic images of number digits resembling the distribution of the MNIST dataset.

## Problem Statement
The task involves creating a model capable of generating high-quality images of handwritten digits, particularly those from 0 to 9, with a focus on ensuring that the generated images are indistinguishable from real handwritten digits.

## Overview of Solutions
The solution involves implementing a DCGAN architecture using TensorFlow, a popular deep learning framework. DCGAN consists of two neural networks: a generator and a discriminator. The generator is responsible for generating synthetic images, while the discriminator tries to distinguish between real and generated images. Through adversarial training, both networks improve iteratively, leading to the generation of realistic images.

## Steps to Implement the Solution

### 1. Data Preparation
- Utilize the MNIST dataset, a collection of 28x28 grayscale images of handwritten digits.
- Preprocess the data, including normalization and reshaping, to prepare it for training.

### 2. Model Architecture
- Design the generator and discriminator networks.
- Generator Network:
  - Input: Random noise vector (latent space).
  - Output: Synthetic images resembling handwritten digits.
- Discriminator Network:
  - Input: Real or generated images.
  - Output: Probability of the input being real.

### 3. Training
- Train the DCGAN model using a combination of real and generated images.
- Alternately train the generator and discriminator networks:
  - Generator aims to produce images that fool the discriminator.
  - Discriminator learns to distinguish between real and generated images.
- Monitor and adjust hyperparameters as necessary to improve performance.

### 4. Evaluation
- Evaluate the quality of generated images using quantitative metrics (e.g., Frechet Inception Distance) and qualitative assessment.
- Assess the ability of the model to produce realistic-looking handwritten digits resembling those in the MNIST dataset.

### 5. Optimization and Fine-Tuning
- Fine-tune the model architecture and training process to improve image quality and convergence speed.
- Experiment with different architectural variations, loss functions, and optimization techniques to enhance performance.

## Repository Structure
- `README.md`: Overview of the project, including problem statement, solution approach, and implementation steps.
- `data/`: Directory for storing the MNIST dataset or other relevant data.
- `src/`: Source code directory containing Python scripts for data preprocessing, model implementation, training, and evaluation.


## Usage
1. Clone the repository to your local machine.
2. Install the necessary dependencies listed in `requirements.txt`.
3. Download the MNIST dataset or provide your own dataset if desired.
4. Run the provided scripts to preprocess the data, train the model, and generate images.
5. Experiment with different parameters and configurations to optimize the model performance.



## Contributors
- PRIYA A


## Demo Link
https://drive.google.com/drive/folders/1Fjij9vftCV-lt0A8gk2SI-6kHnub1jEY
---
