import numpy as np
import torch.utils.data
from matplotlib import pyplot as plt

class Supervised(torch.utils.data.Dataset):
    def __init__(self, patch_size=1, n_channels=24,
                  class_number=0, train=True, test=False, transform=None, ae=False,
                 **kwargs):

        self.n_channels = n_channels
        self.p_size = patch_size
        self.class_number = class_number
        self.transform = transform
        self.ae = ae #not used so far

        # From Kwargs
        if not test:
            self.samples_coords = kwargs['samples_coords_train'] if train else kwargs['samples_coords_val']
            self.labels = kwargs['labels_train'] if train else kwargs['labels_val']
            self.patch_path = kwargs['patch_path_train'] if train else kwargs['patch_path_val']
            self.current_patch_path = None
        else:
            self.samples_coords = kwargs['samples_coords_test']
            self.labels = kwargs['labels_test']
            self.patch_path = kwargs['patch_path_test']
            self.current_patch_path = None

        # self.img_src = self.load_image(self.patch_path).transpose(1, 2, 0)
        # self.img_src = np.clip(self.img_src / 10000, 0, 1)

    def __getitem__(self, index: int) -> (np.array, np.array):
        """
        Return images with shape (C, W, H)
        :param index:
        :return:
        """
        if self.current_patch_path != self.patch_path[index]:
            self.img_src = self.load_image(self.patch_path[index]).transpose(1, 2, 0)
            self.img_src = np.clip(self.img_src / 10000, 0, 1)
            self.current_patch_path = self.patch_path[index]
        else:
            self.current_patch_path = self.patch_path[index]

        batch_coords = self.samples_coords[index]
        x = int(batch_coords.split(' ')[0])
        y = int(batch_coords.split(' ')[1])
        batch = np.zeros((self.p_size, self.p_size, self.n_channels))
        if self.p_size > 1:
            batch[:, :, :] = self.img_src[y - self.p_size // 2: y + self.p_size // 2 + 1
            , x - self.p_size // 2: x + self.p_size // 2 + 1, :]
            labels = np.ones((self.p_size, self.p_size)) * self.labels[index]
        else:
            batch[:, :, :] = self.img_src[y: y + 1, x:  x + 1]
            labels = np.array([self.labels[index]])

        if self.transform is not None:
            images = self.transform(batch)

        return images, labels

    def load_image(self, path, key=None):
        if path.split(".")[-1] == "mat":
            raise NotImplementedError
            #if key == 'X':
            #    w, h = load_mat(path, 'groundtruth').shape
            #    return load_mat(path, key).reshape(-1, w, h).transpose(1, 2, 0)
            #img = load_mat(path, key)
        else:
            img = np.load(path)
        return img

    def __len__(self):
        return len(self.samples_coords)



class Supervised_dictionary(torch.utils.data.Dataset):
    def __init__(self, n_channels=24,
                  class_number=0, train=True, test=False, transform=None, ae=False,
                 **kwargs):

        self.n_channels = n_channels
        self.class_number = class_number
        self.transform = transform

        # From Kwargs
        if not test:
            self.dict = kwargs['train_dict'] if train else kwargs['val_dict']
        else:
            self.dict = kwargs['test_dict']

        if 'patch_size' in kwargs.keys():
            self.p_size = kwargs['patch_size']
        else:
            self.p_size = False

    def __getitem__(self, index: int) -> (np.array, np.array):
        """
        Return images with shape (C, W, H)
        :param index:
        :return:
        """

        if self.p_size:
            batch, labels = self.dict['patches'][index], self.dict['labels'][index]
            central_pixel = (batch.shape[1] // 2)
            if self.p_size > 1:
                batch = batch[central_pixel - self.p_size // 2: central_pixel + self.p_size // 2 + 1
                , central_pixel - self.p_size // 2: central_pixel + self.p_size // 2 + 1, :]
                labels = labels[central_pixel - self.p_size // 2: central_pixel + self.p_size // 2 + 1
                , central_pixel - self.p_size // 2: central_pixel + self.p_size // 2 + 1]
            else:
                batch = batch[central_pixel: central_pixel + 1, central_pixel:  central_pixel + 1]
                labels = labels[central_pixel, central_pixel]

        else:
            batch, labels = self.dict['patches'][index], self.dict['labels'][index]

        batch = np.transpose(batch, (2,0,1))

        return batch, labels

    def __len__(self):
        return len(self.dict['patches'])
