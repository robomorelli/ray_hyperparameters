from preprocessing.nls_kdd_preprocessing import prep_nls_kdd_test, prep_nls_kdd_train_val
from preprocessing.albania_preprocessing import prep_albania
from dataset.nls_kdd_dataloader import numpyArray
from dataset.albania_dataloader import Supervised, Supervised_dictionary

import torch
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader,ConcatDataset
from sklearn.model_selection import KFold
import pickle
import numpy as np
from omegaconf import OmegaConf
import multiprocessing as mp


from config import *


def get_dataset(cfg, **kwargs):
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

    if cfg.dataset.name == "albania_supervised":
        # Preprocessing step (load mixed coords of negative and positive)
        open_file = open(os.path.join(root + cfg.dataset.coords_path), "rb")
        selected_pixels = pickle.load(open_file)
        open_file.close()

        open_file = open(os.path.join(root + cfg.dataset.test_coords_path), "rb")
        test_selected_pixels = pickle.load(open_file)
        open_file.close()

        metrics = ["accuracy", "f1_score"] # evaluate to pass from here the entire metric object to the trainer
        # use case: kwarg cames after loading data a preprocessing step (the can also come from object trainer (cnn3d trainer for example)
        #Kwargs information:

        if not kwargs['from_dictionary']:

            transform = T.Compose([
                T.ToTensor(),
            ])

            c_train, l_train, path_train, c_val, l_val, path_val = prep_albania(selected_pixels, dataset_train_split=cfg.dataset.train_split)
            c_test, l_test, path_test = prep_albania(test_selected_pixels, test=True)

            dataset_train = Supervised(n_channels=cfg.dataset.in_channel, class_number=cfg.model.class_number, train=True,
                                                # From Kwargs:
                                               patch_size=kwargs['patch_size'], batch_size=kwargs['batch_size'],
                                               transform=transform, samples_coords_train=c_train,
                                               labels_train=l_train, patch_path_train=path_train, samples_coords_val=c_val,
                                               labels_val=l_val, patch_path_val=path_val)
            dataset_val = Supervised(n_channels=cfg.dataset.in_channel, class_number=cfg.model.class_number, train=False,
                                                # From Kwargs:
                                               patch_size= kwargs['patch_size'], batch_size = kwargs['batch_size'],
                                               transform=transform, samples_coords_train=c_train,
                                               labels_train=l_train, patch_path_train=path_train, samples_coords_val=c_val,
                                               labels_val=l_val,  patch_path_val=path_val)
            dataset_test = Supervised(#patch_size=cfg.dataset.patch_size
                                                n_channels=cfg.dataset.in_channel, class_number=cfg.model.class_number, train=False,
                                                test=True,
                                                # From Kwargs:
                                                patch_size=kwargs['patch_size'], batch_size = kwargs['batch_size'],
                                                transform=transform, samples_coords_test=c_test,
                                                labels_test=l_test, patch_path_test=path_test)
        else:
            transform = T.Compose([
                T.ToTensor(),
            ])

            train_dict, val_dict = prep_albania(selected_pixels, dataset_train_split=cfg.dataset.train_split,
                                                from_dictionary=cfg.dataset.from_dictionary)

            if kwargs['augmentation']:
                dataset_train = Supervised_dictionary(n_channels=cfg.dataset.in_channel, class_number=cfg.model.class_number, train=True,
                                                       transform=transform,
                                                    # From Kwargs:
                                                    train_dict=train_dict, val_dict=val_dict, patch_size=kwargs['patch_size']
                                                    , augmentation=cfg.opt.augmentation)

            else:
                dataset_train = Supervised_dictionary(n_channels=cfg.dataset.in_channel, class_number=cfg.model.class_number, train=True,
                                                       transform=transform, augmentation=cfg.opt.augmentation,
                                                    # From Kwargs:
                                                    train_dict=train_dict, val_dict=val_dict, patch_size=kwargs['patch_size'])


            dataset_val = Supervised_dictionary(n_channels=cfg.dataset.in_channel, class_number=cfg.model.class_number, train=False,
                                                transform=transform,
                                                # From Kwargs:
                                               patch_size= kwargs['patch_size'],
                                                val_dict=val_dict, train_dict=train_dict)
            dataset_test = Supervised_dictionary(#patch_size=cfg.dataset.patch_size
                                                n_channels=cfg.dataset.in_channel, class_number=cfg.model.class_number, train=False,
                                                test=True, transform=transform,
                                                # From Kwargs:
                                                patch_size=kwargs['patch_size'],
                                                 test_dict=test_selected_pixels)

        if kwargs['oversampling']:

            # kfold = KFold(n_splits=cfg.opt.k_fold_cv, shuffle=True)
            # dataset = ConcatDataset([dataset_train, dataset_val])
            # weights = get_class_weights(dataset_train)
            y = np.array([int(x[1][0][0]) for x in list(dataset_train)])
            class_sample_count = np.array([len(np.where(y == t)[0]) for t in np.unique(y)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in y])

            sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

            train_loader = DataLoader(dataset_train, batch_size=kwargs['batch_size'],
                                      num_workers=1, sampler=sampler)

            val_loader = DataLoader(dataset_val, batch_size=kwargs['batch_size'],
                                    num_workers=1)
            test_loader = DataLoader(dataset_val, batch_size=kwargs['batch_size'],
                                     num_workers=1)
            weights = get_class_weights(dataset_train)

            return train_loader, val_loader, test_loader, weights, metrics

        else:

            train_loader = DataLoader(dataset_train, batch_size=kwargs['batch_size'],
                                      num_workers=cfg.opt.num_workers, shuffle=True)  # num_workers=num_workers,
            val_loader = DataLoader(dataset_val, batch_size=kwargs['batch_size'],
                                    num_workers=cfg.opt.num_workers, shuffle=False)  # num_workers=num_workers,
            test_loader = DataLoader(dataset_test, batch_size=kwargs['batch_size'],
                                     num_workers=cfg.opt.num_workers, shuffle=False)  # num_workers=num_workers,
            weights = get_class_weights(dataset_train)

            return train_loader, val_loader, test_loader, weights, metrics



def get_class_weights(dataset):
    if dataset.class_number > 0:
        class_frequency = class_frequency_center_pixel(list(dataset), dataset.class_number)
        weights = [1 / (x / np.sum(class_frequency)) for x in class_frequency]
        weights = torch.FloatTensor(weights / np.max(weights))
        if dataset.class_number == 1:
            pos_weight = (weights[1]/weights[0])
            print(f"Class frequency: {class_frequency}")
            print(f"Class weights: {weights}")
            print(f"reweight: {pos_weight}")
            return pos_weight
        print(f"Class frequency: {class_frequency}")
        print(f"Class weights: {weights}")
        return weights
    else:
        return None

def class_frequency_center_pixel(patch_list, n_classes) -> np.array:
    """
    Compute the frequency of the central pixel
    :param patch_list: list of patches
    :return: array of the same length of the classes with their relative frequency
    """
    print("Computing class frequency...")
    target_list = []
    for patch in patch_list:
        target = patch[1]
        if target.shape[0] > 1:
            w = int(target.shape[0] / 2)
            h = int(target.shape[1] / 2)
            central_pixel = target[w, h]
        else:
            central_pixel = target[0]
        target_list.append(int(central_pixel))

    if n_classes > 1:
        target_arr = np.ones(n_classes)
    elif n_classes == 1:
        target_arr = np.ones(n_classes+1)

    for t in target_list:
        target_arr[t] += 1

    return target_arr

#if __name__ == "__main__":
#    cfg = OmegaConf.load("/home/roberto/Documents/backup_rob/esa/fdir/train_configurations/cnn3d" + '.yaml')
#    get_dataset(cfg)