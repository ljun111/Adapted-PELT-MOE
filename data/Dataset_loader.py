import numpy as np
from torch.utils.data import Dataset, DataLoader

class NpyDataset(Dataset):
    def __init__(self, npy_file):
        self.data = np.load(npy_file, allow_pickle=True)
        self.data = np.array(self.data, dtype=np.float64)
        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        normalized_sample = (sample - self.mean) / (self.std)
        return normalized_sample
