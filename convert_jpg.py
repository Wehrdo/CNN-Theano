import numpy as np
import matplotlib.image
import os
import pickle

import cifar_utils

def convert_dir(dir_name):
    parsed_images = []
    for fname in os.listdir(dir_name):
        im_mat = matplotlib.image.imread(os.path.join(dir_name, fname))[:,:,0:3]
        flattened = np.ravel(im_mat)
        parsed_images.append(flattened)
    return np.vstack(parsed_images)

if __name__ == '__main__':
    buses = convert_dir('datasets/only_buses')
    no_buses = convert_dir('datasets/no_buses')
    labels = np.hstack((np.repeat(1, buses.shape[0]),
                       np.repeat(0, no_buses.shape[0]))).astype('int32')
    all_data = np.vstack((buses, no_buses)).astype('float32')
    ids = list(range(labels.shape[0]))
    np.random.shuffle(ids)
    test_set_size = 150
    test_indices = ids[:test_set_size]
    train_indices = ids[test_set_size:]
    test_set = (all_data[test_indices], labels[test_indices])
    train_set = (all_data[train_indices], labels[train_indices])

    to_write = (train_set, test_set, None)
    with open(f'datasets/combined_buses.pkl3', 'wb') as file:
        pickle.dump(to_write, file),
