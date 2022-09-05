#!/usr/bin/env python3
from torch.utils.data import DataLoader
from config import *
from trainer.vae_trainer import trainVae
from trainer.ae_trainer import trainAe
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
