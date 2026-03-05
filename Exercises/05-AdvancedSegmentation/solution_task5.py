import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.measure import label
from scipy import ndimage as ndi

def apply_otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    Computes the Otsu threshold and returns a binary mask.
    """
    thresh = threshold_otsu(image)
    binary = image > thresh
    return binary.astype(np.uint8)

def apply_watershed(image: np.ndarray) -> np.ndarray:
    """
    Applies the watershed algorithm to segment individual objects.
    """
    # Thresholding
    thresh = threshold_otsu(image)
    binary = image > thresh
    
    # Distance transform
    distance = ndi.distance_transform_edt(binary)
    
    # Find local maxima as markers
    # Simple version: find non-zero distance pixels and label them
    from skimage.feature import peak_local_max
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    
    # Watershed
    labels = watershed(-distance, markers, mask=binary)
    return labels

class SimpleCNN(nn.Module):
    def __init__(self):
        """
        A very simple CNN for binary segmentation.
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device):
    """
    Standard PyTorch training loop for one epoch.
    """
    model.train()
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """
    Evaluates the model on a dataset and returns the mean IoU score.
    """
    model.eval()
    ious = []
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            
            # Simple IoU calculation
            preds = (torch.sigmoid(outputs) > 0.5).float()
            intersection = (preds * targets).sum(dim=(1, 2, 3))
            union = (preds + targets).clamp(0, 1).sum(dim=(1, 2, 3))
            
            iou = (intersection + 1e-6) / (union + 1e-6)
            ious.extend(iou.cpu().numpy())
            
    return float(np.mean(ious))
