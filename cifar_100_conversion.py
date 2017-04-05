import numpy as np
import theano.tensor as T
from theano import function as Tfunc
import pickle

all_data = []
all_labels = []


with open('datasets/cifar-100-python/train', 'rb') as file:
    data = pickle.load(file, encoding='latin1')
    all_labels.extend(data['fine_labels'])
    all_data.append(data['data'])

combined_x = np.vstack(tuple(all_data))
combined_y = np.array(all_labels).reshape(-1,)
print(combined_x.shape, combined_y.shape)

with open(f'datasets/cifar-100-python/test', 'rb') as file:
    data = pickle.load(file, encoding='latin1')
    test_x = data['data']
    test_y = np.array(data['fine_labels']).reshape(-1,)

print(test_x.shape, test_y.shape)

combined_x = combined_x.astype('float32') / 255
test_x = test_x.astype('float32') / 255


with open(f'datasets/cifar-100-python/combined_data.pkl3', 'wb') as file:
    pickle.dump(((combined_x, combined_y), (test_x, test_y), None), file),

#dict_keys(['batch_label', 'labels', 'data', 'filenames'])