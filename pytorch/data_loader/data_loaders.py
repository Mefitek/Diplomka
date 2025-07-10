import os
import torch
from torch.utils.data import Dataset
from base import BaseDataLoader

# =====================================================
# =========       Dataloaders for NNs         =========
# =====================================================

class PTDataset(Dataset):
    """
    Loads preprocessed time-series data from a .pt file.
    The .pt file must contain a dict with keys 'inputs' and 'labels'.
    """
    def __init__(self, pt_path, augment=False, flip_prob=0.5, noise_std=0.01):
        data = torch.load(pt_path)
        self.inputs = data['inputs']   # shape: [N, C, T]
        self.labels = data['labels']   # shape: [N]
        self.augment = augment
        self.flip_prob = flip_prob
        self.noise_std = noise_std

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        #x = self.inputs[idx].clone()
        x = self.inputs[idx]
        y = self.labels[idx]

        if self.augment:
            # Random horizontal flip (time reversal)
            if torch.rand(1).item() < self.flip_prob:
                x = torch.flip(x, dims=[-1])  # Flip across time axis

            # Add Gaussian noise
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        return x, y

class PTDataLoader(BaseDataLoader):
    """
    DataLoader wrapper for a preprocessed PTDataset.
    Selects one of: train / val / test based on `training` flag and file availability.
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=2,
                 training=True, test=False, augment=False, flip_prob=0.5, noise_std=0.01):
        if test:
            filename = 'dataset_test.pt'
        elif training:
            filename = 'dataset_train.pt'
        else:
            filename = 'dataset_val.pt'

        pt_path = os.path.join(data_dir, filename)
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Dataset file not found: {pt_path}")

        do_augment = augment and training # augment only in training
        dataset = PTDataset(
            pt_path,
            augment=do_augment,  
            flip_prob=flip_prob,
            noise_std=noise_std
        )
        super().__init__(dataset, batch_size, shuffle, validation_split=0.0, num_workers=num_workers)

# =============================================
# =========       Autoencoder         =========
# =============================================

class AE_PTDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path)
        self.inputs = data['inputs']   # shape: [N, C, T]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        return x, x  # input == target

class AE_PTDataset_test(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path)
        self.inputs = data['inputs']   # shape: [N, C, T]
        self.labels = data['labels']

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.labels[idx]  # binary channel
        return x, y

class AE_PTDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=2, training=True, test=False):
        if test:
            filename = 'dataset_test.pt'
        elif training:
            filename = 'dataset_train.pt'
        else:
            filename = 'dataset_val.pt'
        pt_path = os.path.join(data_dir, filename)
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Dataset file not found: {pt_path}")

        if test:
            dataset = AE_PTDataset_test(pt_path)
        else:
            dataset = AE_PTDataset(pt_path)
        super().__init__(dataset, batch_size, shuffle, validation_split=0.0, num_workers=num_workers)

