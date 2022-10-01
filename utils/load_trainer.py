#!/usr/bin/env python3
from torch.utils.data import DataLoader
from trainer.vae_trainer import trainVae
from trainer.ae_trainer import trainAe
from trainer.cnn3d_trainer import trainCNN3D
from torch.utils.data import Dataset

def get_trainer(cfg, **kwargs):
    """
    Get the dataset.
    :param cfg:  configuration file
    :param transform: transform to be applied to the dataset
    :return: dataset train, dataset test
    """
    if cfg.model.name == "vae":
        return trainVae
    if cfg.model.name == "ae":
        return trainAe
    if cfg.model.name == "cnn3d":
        return trainCNN3D