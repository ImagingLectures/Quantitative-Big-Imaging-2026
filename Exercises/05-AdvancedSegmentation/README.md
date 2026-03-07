# Exercises for lecture 5 - Advanced segmentation

## Learning Objectives
- Explore and evaluate simple segmentation algorithms (Otsu, Watershed)
- Implement, train, and evaluate a small CNN for binary segmentation

## Preparation
- Run `bash setup.sh` from the terminal in the exercise directory. You should see the python environment getting updated.

# Exercise
The notebooks `segmentation_intro.ipynb` and `cnn_segmentation_intro.ipynb` introduce advanced segmentation techniques and deep learning based approaches.

In `tasks.py`, you will implement the following:

### 1. Basic Segmentation
- **Otsu Thresholding**: Implement `apply_otsu_threshold` to automatically find the best threshold for binarizing an image.
- **Watershed Segmentation**: Implement `apply_watershed` to separate individual objects in an image.

### 2. CNN Segmentation
- **SimpleCNN**: Define a small convolutional neural network for pixel-wise binary classification.
- **Training Loop**: Implement `train_one_epoch` to train your model on a dataset.
- **Evaluation**: Implement `evaluate_model` to measure the performance using the IoU metric.

## Evaluation
You can verify your implementation by running the provided tests:
```bash
uv run pytest test_exercise5.py
```
First, ensure all tests pass (they will initially fail with `NotImplementedError`). Your goal is to fill in the code in `tasks.py` until all tests in `test_exercise5.py` pass.
