from preprocessing.nls_kdd_preprocessing import prep_nls_kdd_test, prep_nls_kdd_train_val
from dataset.nls_kdd_dataloader import numpyArray
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from config import *

def get_dataset(cfg, transform=None, **kwargs):
    """
    Get the dataset.
    :param cfg:  configuration file
    :param transform: transform to be applied to the dataset
    :return: dataset train, dataset test
    """
    if cfg.dataset.name == "nls_kdd":
        # Preprocessing step (there is also testx to use for example in test accuray in trainer step
        trainx, valx, ohe = prep_nls_kdd_train_val(0.2, nls_kdd_cols, nls_kdd_cat_cols,
                                          preprocessing='log', exclude_cat=True)
        # Dataset for dataloader definition
        dataset_train = numpyArray(trainx)
        dataset_val = numpyArray(valx)
        # Dataloader definition
        trainloader = DataLoader(dataset_train, batch_size=kwargs['batch_size'], shuffle=True)
        valloader = DataLoader(dataset_val, batch_size=kwargs['batch_size'], shuffle=True)

        return trainloader, valloader, len(nls_kdd_cols), len(nls_kdd_cat_cols), trainx.shape[1]
