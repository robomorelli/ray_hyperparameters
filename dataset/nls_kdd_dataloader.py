import torch
from torch.utils.data import Dataset

class Numpy_array(Dataset):
    def __init__(self, matrix, autoencoder=True):
        self.autoencoder = autoencoder
        self.x = matrix
        self.X = torch.tensor(self.x, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.autoencoder:
            return self.X[idx], self.X[idx]
        else:
            raise NotImplementedError

class pandasDF(Dataset):
    def __init__(self, df, autoencoder=True):
        self.autoencoder = autoencoder
        self.x = df.values
        self.X = torch.tensor(self.x, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.autoencoder:
            return self.X[idx], self.X[idx]
        else:
            raise NotImplementedError