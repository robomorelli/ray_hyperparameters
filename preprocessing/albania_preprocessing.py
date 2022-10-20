
import numpy as np
import random
from sklearn.model_selection import train_test_split

def prep_albania(selected_pixels, dataset_train_split=0.70, test=False, from_dictionary=False):

    if not from_dictionary:
        if not test:
            random.shuffle(selected_pixels)
            train_size = int(np.ceil(len(selected_pixels) * dataset_train_split))
            selected_pixels_train = selected_pixels[:train_size]
            selected_pixels_val = selected_pixels[train_size:]

            pixels_coords_train = selected_pixels_train
            samples_coords_train = [' '.join((str(x[0]), str(x[1]))) for x in pixels_coords_train]
            labels_train = [1 if x[-2] == 'g' else 0 for x in pixels_coords_train]
            patch_path_train = [x[-1] for x in pixels_coords_train]

            pixels_coords_val = selected_pixels_val
            samples_coords_val = [' '.join((str(x[0]), str(x[1]))) for x in pixels_coords_val]
            labels_val = [1 if x[-2] == 'g' else 0 for x in pixels_coords_val]
            patch_path_val = [x[-1] for x in pixels_coords_val]

            return samples_coords_train, labels_train, patch_path_train, samples_coords_val, labels_val, patch_path_val
        else:
            random.shuffle(selected_pixels)
            pixels_coords_test = selected_pixels
            samples_coords_test = [' '.join((str(x[0]), str(x[1]))) for x in pixels_coords_test]
            labels_test= [1 if x[-2] == 'g' else 0 for x in pixels_coords_test]
            patch_path_test = [x[-1] for x in pixels_coords_test]

            return samples_coords_test, labels_test, patch_path_test

    else:
        if not test:
            X_train, X_test, y_train, y_test = train_test_split(selected_pixels['patches'],
                                                                selected_pixels['labels'],
                                                                test_size=1 - dataset_train_split)
            train_dict = {}
            val_dict = {}
            train_dict['patches'] = X_train
            train_dict['labels'] = y_train
            val_dict['patches'] = X_test
            val_dict['labels'] = y_test
            return train_dict, val_dict

        else:
            X_train, X_test, y_train, y_test = train_test_split(selected_pixels['patches'],
                                                                selected_pixels['labels'],
                                                                test_size=1 - dataset_train_split)
            train_dict = {}
            val_dict = {}
            train_dict['patches'] = X_train
            train_dict['labels'] = y_train
            val_dict['patches'] = X_test
            val_dict['labels'] = y_test

            return val_dict
