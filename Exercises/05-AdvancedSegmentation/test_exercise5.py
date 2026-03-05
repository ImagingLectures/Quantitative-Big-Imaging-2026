import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tasks import *

def test_apply_otsu_threshold():
    # Create simple image with two distinct intensities
    image = np.zeros((10, 10))
    image[:5, :] = 100
    image[5:, :] = 200
    
    mask = apply_otsu_threshold(image)
    assert mask.shape == image.shape
    assert np.all(np.logical_or(mask == 0, mask == 1))
    assert np.mean(mask[:5, :]) == 0
    assert np.mean(mask[5:, :]) == 1

def test_apply_watershed():
    # Create two circles in an image
    image = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            if (i-5)**2 + (j-5)**2 < 9:
                image[i, j] = 255
            if (i-15)**2 + (j-15)**2 < 9:
                image[i, j] = 255
    
    labeled = apply_watershed(image)
    assert labeled.shape == image.shape
    assert len(np.unique(labeled)) >= 3  # background + 2 objects

def test_simple_cnn_init():
    model = SimpleCNN()
    assert isinstance(model, nn.Module)

def test_simple_cnn_forward():
    model = SimpleCNN()
    dummy_input = torch.randn(1, 1, 32, 32)
    output = model(dummy_input)
    assert output.shape == (1, 1, 32, 32)

def test_train_one_epoch():
    model = SimpleCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cpu")
    
    # Dummy data
    x = torch.randn(4, 1, 32, 32)
    y = torch.randint(0, 2, (4, 1, 32, 32)).float()
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=2)
    
    initial_params = [p.clone() for p in model.parameters()]
    train_one_epoch(model, dataloader, optimizer, criterion, device)
    
    # Check that parameters updated
    for p_initial, p_after in zip(initial_params, model.parameters()):
        assert not torch.equal(p_initial, p_after)

def test_evaluate_model():
    model = SimpleCNN()
    device = torch.device("cpu")
    
    # Dummy data
    x = torch.randn(2, 1, 32, 32)
    y = torch.randint(0, 2, (2, 1, 32, 32)).float()
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=2)
    
    iou = evaluate_model(model, dataloader, device)
    assert isinstance(iou, float)
    assert 0.0 <= iou <= 1.0
