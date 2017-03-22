import pickle
import numpy as np

def load_data(path, amt=-1):
    with open(path, 'rb') as data_f:
        train_set, test_set, validation_set = pickle.load(data_f)

    train_data = train_set[0][0:amt]
    train_labels = train_set[1][0:amt]

    transform = preprocess(train_data)

    train_x = cifar_to_im(train_data)

    n_classes = len(np.unique(train_labels))
    train_y = transform_labels(train_labels, n_classes)

    test_y = transform_labels(test_set[1], 10)
    test_data = test_set[0]
    preprocess(test_data, transform)
    test_x = cifar_to_im(test_data)

    return (train_x, train_y, test_x, test_y)

def cifar_to_im(dataset):
    n_images = dataset.shape[0]
    return np.transpose(dataset.T.reshape((32,32,3,n_images), order='F'), [3,2,0,1]).astype('float32')

def transform_labels(labels, n_classes=None):
    if n_classes is None:
        n_classes = len(np.unique(labels))
    y = np.zeros((n_classes, labels.shape[0]))
    for c in range(n_classes):
        y[c, np.where(labels == c)[0]] = 1
    return np.transpose(y).astype('float32')


def preprocess(data, params=None):
    if params == None:
        mean = np.mean(data, axis=0)
        data -= mean
        stdev = np.std(data, axis=0)
        data /= stdev
        return (mean, stdev)
    else:
        data -= params[0]
        data /= params[1]
        return params