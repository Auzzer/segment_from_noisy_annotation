"""Dataset utilities for NIfTI medical image segmentation."""

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from pathlib import Path


def _strip_extensions(name: str) -> str:
    """Remove chained extensions like .nii.gz."""
    prev = name
    while True:
        stem = Path(prev).stem
        if stem == prev or stem == "":
            return prev
        prev = stem


def _normalize_image_id(name: str) -> str:
    """Normalize various id representations to 'image_XXX' format."""
    base = _strip_extensions(name)
    if base.startswith('image_'):
        return base
    if base.startswith('label_'):
        suffix = base.split('_', 1)[1]
        return f'image_{suffix}'
    if base.isdigit():
        return f'image_{int(base):03d}'
    return base


def _load_id_list(list_path):
    with open(list_path, 'r') as f:
        lines = [_normalize_image_id(line.strip()) for line in f if line.strip()]
    return lines


class NiftiSegmentationDataset(Dataset):
    """Dataset for loading NIfTI medical imaging data for segmentation."""
    
    def __init__(self, data_dir, transform=None, patch_size=(64, 64, 64),
                 normalize=True, allowed_ids=None):
        """
        Initialize NIfTI segmentation dataset.
        
        Args:
            data_dir: Directory containing 'images' and 'labels' subdirectories
            transform: Optional transform to be applied on a sample
            patch_size: Size of the patches to extract (D, H, W)
            normalize: Whether to normalize the images
            allowed_ids: Optional list of image base names to include
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images'
        self.labels_dir = self.data_dir / 'labels'
        self.transform = transform
        self.patch_size = patch_size
        self.normalize = normalize
        self.allowed_ids = set(allowed_ids) if allowed_ids is not None else None
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob('*.nii.gz')))
        
        # Filter to only include images that have corresponding labels
        self.valid_samples = []
        for img_file in self.image_files:
            img_id = _normalize_image_id(img_file.name)
            # Extract numeric suffix for label lookup
            suffix = img_id.split('_', 1)[1] if '_' in img_id else img_id
            label_file = self.labels_dir / f'label_{suffix}.nii.gz'
            if label_file.exists():
                if self.allowed_ids is None or img_id in self.allowed_ids:
                    self.valid_samples.append((img_file, label_file))
            else:
                if self.allowed_ids and img_id in self.allowed_ids:
                    print(f"Warning: label missing for {img_file.name}; skipping.")

        print(f"Found {len(self.valid_samples)} valid image-label pairs")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        img_path, label_path = self.valid_samples[idx]
        
        # Load NIfTI files
        img_nifti = nib.load(str(img_path))
        label_nifti = nib.load(str(label_path))
        
        # Get data as numpy arrays
        image = img_nifti.get_fdata().astype(np.float32)
        label = label_nifti.get_fdata().astype(np.float32)
        
        # Normalize image
        if self.normalize:
            image = (image - image.mean()) / (image.std() + 1e-8)
        
        # Extract random patch
        image, label = self._extract_patch(image, label)
        
        # Add channel dimension
        image = np.expand_dims(image, axis=0)  # (1, D, H, W)
        label = np.expand_dims(label, axis=0)  # (1, D, H, W)
        
        # Convert to torch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _extract_patch(self, image, label):
        """
        Extract a random patch from the image and label.
        
        Args:
            image: Input 3D image array
            label: Input 3D label array
            
        Returns:
            Tuple of (image_patch, label_patch)
        """
        d, h, w = image.shape
        pd, ph, pw = self.patch_size
        
        # If image is smaller than patch size, pad it
        if d < pd or h < ph or w < pw:
            pad_d = max(0, pd - d)
            pad_h = max(0, ph - h)
            pad_w = max(0, pw - w)
            
            image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            label = np.pad(label, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            d, h, w = image.shape
        
        # Random crop
        d_start = np.random.randint(0, d - pd + 1) if d > pd else 0
        h_start = np.random.randint(0, h - ph + 1) if h > ph else 0
        w_start = np.random.randint(0, w - pw + 1) if w > pw else 0
        
        image_patch = image[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        label_patch = label[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        
        return image_patch, label_patch


def get_train_val_dataloaders(data_dir, batch_size=2, val_split=0.2, num_workers=4,
                              patch_size=(64, 64, 64), train_list_path=None):
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
        num_workers: Number of workers for data loading
        patch_size: Size of patches to extract
        train_list_path: Optional path to file with allowed image ids
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Subset
    
    allowed_ids = None
    if train_list_path:
        allowed_ids = _load_id_list(train_list_path)
        print(f"Restricting training dataset to {len(allowed_ids)} ids from {train_list_path}")
    
    # Create full dataset
    full_dataset = NiftiSegmentationDataset(
        data_dir,
        patch_size=patch_size,
        allowed_ids=allowed_ids
    )
    
    dataset_size = len(full_dataset)
    if dataset_size == 0:
        raise RuntimeError("No samples available for training after applying filters.")
    
    # Split into train and validation
    indices = list(range(dataset_size))
    
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1 (exclusive).")
    
    train_indices, val_indices = train_test_split(
        indices, test_size=val_split, random_state=42
    )
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
