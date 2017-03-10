from typing import Tuple, List

import matplotlib
import numpy as np
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import function as Tfunc
from theano import shared
from theano.tensor.nnet.bn import batch_normalization
import theano
import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.colors
import sys
import math
import time
import pickle
import json

MARGIN = 1

class MLP:
    def __init__(self, layers, dropout_rate, trained_model=None):
        n_layers = len(layers)
        activation_sizes = []
        if trained_model is None:
            self.weights = []
            self.biases = []
            self.gammas = []
            self.betas = []
            # Initialize random values for weight and biase matrices
            for i in range(1, n_layers):
                n_kernels = layers[i].dims[0]
                # kernel_size = layer_sizes[i-1][0] - layer_sizes[i][0] + 1
                kernel_size = layers[i].kernel_size
                w_mat_size = (n_kernels, layers[i-1].dims[0], kernel_size, kernel_size)

                w = np.random.standard_normal(w_mat_size).astype('float32') * 0.001#math.sqrt(2 / w_mat_size[1])
                b = np.random.standard_normal((1,n_kernels,1,1)).astype('float32') * 0.001#math.sqrt(2 / w_mat_size[1])

                # w = np.random.standard_normal((layer_sizes[i], layer_sizes[i-1])) * math.sqrt(2 / layer_sizes[i-1])
                # b = np.random.standard_normal((layer_sizes[i], 1)) * math.sqrt(2 / layer_sizes[i-1])
                self.weights.append(w)
                self.biases.append(b)

                activation_size = [layers[i].dims[0], layers[i].dims[1], layers[i].dims[2]]
                if layers[i].pool is not None:
                    activation_size[1] *= layers[i].pool
                    activation_size[2] *= layers[i].pool
                activation_sizes.append(activation_size)
                if i != n_layers - 1:
                    self.gammas.append(shared(np.ones((activation_size), dtype=theano.config.floatX), f"gamma{i}"))
                    self.betas.append(shared(np.zeros((activation_size), dtype=theano.config.floatX), f"beta{i}"))
        else:
            self.weights = trained_model[0]
            self.biases = trained_model[1]

        #### Theano definitions ####

        # Turn weights and biases into shared variables to be used in training and predicting
        self.weights = [shared(w, f"w{i}") for i, w in enumerate(self.weights)]
        self.biases = [shared(b, f"b{i}", broadcastable=(True,False,True,True)) for i, b in enumerate(self.biases)]

        # Variable for y value (correct values when training)
        y = T.matrix('y')

        # List of matrices for each layer's activations, plus one for input
        activations = T.tensor4s(n_layers)

        images = T.tensor4('images')
        activations[0] = images #im2col.im2col(images, kernel_sizes[0])

        # Dropout random number matrices
        dropout_masks = []
        np.random.seed(1)
        np_rng = np.random.RandomState()
        for i in range(1, n_layers):
            srng = RandomStreams(np_rng.randint(1000000))
            rv = srng.uniform((y.shape[0], *activation_sizes[i-1]))
            dropout_masks.append(rv)
        for l_i in range(1, n_layers):
            # Define layer function: max(0, w*x + b)
            convolved = T.nnet.conv2d(activations[l_i-1], self.weights[l_i-1], border_mode=layers[l_i].mode) + self.biases[l_i-1]
            if l_i != 1:
                dropout_mask = (dropout_masks[l_i-1] < dropout_rate) / dropout_rate
                convolved = convolved * dropout_mask
            if l_i != n_layers - 1:
                convolved = batch_normalization(convolved, self.gammas[l_i-1], self.betas[l_i-1],
                                                T.mean(convolved, axis=(0,), dtype=theano.config.floatX, keepdims=True),
                                                T.std(convolved, axis=(0,), keepdims=True))
                convolved = T.nnet.relu(convolved)
            if layers[l_i].pool is not None:
                pool_size = layers[l_i].pool
                convolved = T.signal.pool.pool_2d(convolved, (pool_size, pool_size), ignore_border=False)
            activations[l_i] = convolved
                # activation = T.nnet.relu(self.biases[l_i] + T.tensordot(self.weights[l_i], columnized, axes=1))
                # activations[l_i + 1] = im2col.col2im(activation, layer_sizes[l_i + 1])
            # elif l_i == 0:
            #     activations[l_i + 1] = T.nnet.relu(self.biases[l_i] + T.dot(self.weights[l_i], activations[l_i]))

        # reshaped_output = im2col.col2im(activations[-1], (layer_sizes[-1][0], layer_sizes[-1][1]), layer_sizes[-1][2])
        #
        # ## Hinge loss
        # # Indices of the correct classes
        # correct_classes = T.argmax(y[0,0], axis=0, keepdims=True)
        # # Actual values predicted for the correct classes
        # correct_vals = reshaped_output[0,0,correct_classes, T.arange(y.shape[3])]
        # # margin: max(0, predicted - correct_value + 1)
        # margin_mat = T.maximum(0, reshaped_output[0,0] - T.repeat(correct_vals, repeats=y.shape[2], axis=0) + 1)
        # # Loss for each training sample
        # individual_losses = T.sum(margin_mat, axis=0) - margin_mat[correct_classes, T.arange(y.shape[3])]
        # tot_loss = T.sum(individual_losses)
        # # Average loss
        # loss = tot_loss / y.shape[3]

        # TODO: Update loss to work with imagized version
        n_classes = y.shape[1]
        n_samples = y.shape[0]
        # activations[-1] is (C,1,N). Drop the middle dimension, then transpose
        reshaped_x = activations[-1].reshape((n_samples, n_classes))
        individual_losses = T.nnet.categorical_crossentropy(T.nnet.softmax(reshaped_x), y)
        loss = T.sum(individual_losses) / T.cast(n_samples, 'float32')

        # List of matrices for each layer's activations, plus one for input
        pred_activations = T.tensor4s(n_layers)
        pred_images = T.tensor4('pred_images')
        pred_activations[0] = pred_images #im2col.im2col(pred_images, kernel_sizes[0])
        for l_i in range(1, n_layers):
            # Define layer function: max(0, w*x + b)
            convolved = T.nnet.conv2d(pred_activations[l_i-1], self.weights[l_i-1], border_mode=layers[l_i].mode) + self.biases[l_i-1]
            if l_i != n_layers - 1:
                this_std = T.std(convolved, axis=(0,), keepdims=True)
                convolved = batch_normalization(convolved, self.gammas[l_i-1] / this_std, self.betas[l_i-1],
                                                T.mean(convolved, axis=(0,), dtype=theano.config.floatX, keepdims=True),
                                                T.ones_like(T.var(convolved, axis=(0,), keepdims=True)))
                convolved = T.nnet.relu(convolved)
            if layers[l_i].pool is not None:
                pool_size = layers[l_i].pool
                convolved = T.signal.pool.pool_2d(convolved, (pool_size, pool_size), ignore_border=False)
            pred_activations[l_i] = convolved
        # pred_output = pred_activations[-1]
        # Create Theano function for predicting
        # self.predict = Tfunc([pred_images], pred_output)
        self.predict = Tfunc([pred_images], pred_activations[-1])

        # List of expressions for derivatives: d_w1, d_w2, ... d_b1, d_b2,...
        derivatives = T.grad(loss, self.weights + self.biases + self.gammas + self.betas)
        # Learning rate
        rate = T.scalar('r')

        # How to update weights and biases when training
        update_rules = self.adam_update(rate, derivatives, self.weights + self.biases + self.gammas + self.betas)
        # update_rules = [(var, var - rate*d_var) for var, d_var in zip(self.weights + self.biases, derivatives)]

        # Function for actually executing training
        self.update_step = Tfunc([images, y, rate], loss, updates=update_rules)

    def adam_update(self, rate, derivatives, values):
        B1 = np.float32(0.9)
        B2 = np.float32(0.999)
        eps = np.float32(1e-8)
        decayed_B1 = shared(B1, 'B1')
        decayed_B2 = shared(B2, 'B2')
        update_rules = []
        for i, param in enumerate(values):
            param_dims = param.get_value(borrow=True).shape
            moment1 = shared(np.zeros(param_dims).astype('float32'), broadcastable=param.broadcastable)
            moment2 = shared(np.zeros(param_dims).astype('float32'), broadcastable=param.broadcastable)
            # computation
            gradient = derivatives[i]
            new_moment1 = (B1 * moment1) + ((np.float32(1) - B1) * gradient)
            new_moment2 = (B2 * moment2) + ((np.float32(1) - B2) * gradient * gradient)
            moment1_est = new_moment1 / (np.float32(1) - decayed_B1)
            moment2_est = new_moment2 / (np.float32(1) - decayed_B2)
            param_update = param - rate*(moment1_est / (T.sqrt(moment2_est) + eps))
            update_rules.append((param, param_update))
            update_rules.append((moment1, new_moment1))
            update_rules.append((moment2, new_moment2))
        update_rules.append((decayed_B1, B1 * decayed_B1))
        update_rules.append((decayed_B2, B2 * decayed_B2))

        return update_rules


    def train(self, x, y, test_x, test_y, rate=0.001, batch_size=200, epochs=30):
        losses = []
        train_accuracies = []
        test_accuracies = []
        for epoch in range(epochs):
            n_iters = int(x.shape[0] / batch_size)
            for iter in range(n_iters):
                selection = np.random.randint(0, x.shape[0], batch_size)
                # loss = self.update_weights(x[:,selection], y[:,selection], rate)
                loss = self.update_step(x[selection,:,:,:], y[selection,:], rate)
                losses.append(loss)
            train_accuracies.append(calc_accuracy(self, x, y, batch_size))
            test_accuracies.append(calc_accuracy(self, test_x, test_y, batch_size))
            print(f"Epoch {epoch}, train: {train_accuracies[-1]}, test: {test_accuracies[-1]}")
            rate *= 0.95
        return losses, train_accuracies, test_accuracies

class Layer:
    def __init__(self, dims: Tuple, kernel_size, pool=None, mode='valid', width=None, height=None):
        self.dims = dims
        self.kernel_size = kernel_size
        self.pool = pool
        self.mode = mode



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

def calc_accuracy(mlp, data_x, data_y, pred_size = 100):
    output = np.empty((data_y.shape[0], data_y.shape[1], 1, 1))
    n_predictions = int(data_y.shape[0] / pred_size)
    for i in range(n_predictions):
        start_lim = i * pred_size
        end_lim = (i+1) * pred_size
        output[start_lim:end_lim] = mlp.predict(data_x[start_lim:end_lim])
    predictions = np.argmax(output[:,:,0,0], axis=1)
    accuracy = np.sum(np.argmax(data_y, axis=1) == predictions) / data_y.shape[0]
    return accuracy

def transform_labels(labels, n_classes=None):
    if n_classes is None:
        n_classes = len(np.unique(labels))
    y = np.zeros((n_classes, labels.shape[0]))
    for c in range(n_classes):
        y[c, np.where(labels == c)[0]] = 1
    return np.transpose(y).astype('float32')

def show_predictions(data_x, data_predictions, class_names={}, n_to_show=50):
    n_rows = 8
    n_cols = int(n_to_show / n_rows)
    gspec = matplotlib.gridspec.GridSpec(n_rows, n_cols)
    # gspec.update(wspace=0.05, hspace=0.05)
    scale_fac = data_x.max() - data_x.min()
    data_x_norm = (data_x - data_x.min()) * (1/scale_fac)
    for row in range(n_rows):
        for col in range(n_cols):
            rand_img = np.random.randint(0, data_x.shape[1])
            img_idx = row*n_cols + col
            ax = plt.subplot(gspec[img_idx])
            ax.imshow(np.transpose(data_x_norm[:,rand_img].reshape((32,32,3), order='F'), [1,0,2]))
            ax.set_title(class_names.get(data_predictions[rand_img], str(data_predictions[rand_img])))
            ax.axis('off')
    plt.show()

def cifar_to_im(dataset):
    n_images = dataset.shape[0]
    return np.transpose(dataset.T.reshape((32,32,3,n_images), order='F'), [3,2,0,1]).astype('float32')

if __name__ == '__main__':
    with open('comp_config.json') as f:
        comp_config = json.load(f)
    # with open('datasets/mnist.pkl3', 'rb') as data_f:
    with open(comp_config['dataset'], 'rb') as data_f:
        train_set, test_set, validation_set = pickle.load(data_f)
    # with gzip.open('datasets/mnist.pkl3.gz', 'wb') as data_f:
    #     pickle.dump((train_set, test_set, validation_set), data_f)

    amt = 50000
    train_data = train_set[0][0:amt]
    train_labels = train_set[1][0:amt]

    transform = preprocess(train_data)

    train_x = cifar_to_im(train_data)

    # n_classes = len(np.unique(train_labels))
    train_y = transform_labels(train_labels, 10)

    test_y = transform_labels(test_set[1], 10)
    test_data = test_set[0]
    preprocess(test_data, transform)
    test_x = cifar_to_im(test_data)
    train_scratch = True
    layers = [Layer((3,32,32), 0), Layer((64,16,16), 5, pool=2, mode='half'), Layer((32,8,8), 3, pool=2, mode='half'), Layer((64,1,1), 8), Layer((10,1,1), 1)]
    # layers = [Layer(3, 0), Layer(32, 3), Layer(10, 30)]
    # layers = [Layer(3, 0), Layer(70, 32), Layer(10, 1)]
    dropout = 0.7
    batch = 100
    epochs = 25
    # alpha = 0.1
    alpha = 0.001

    if train_scratch:
        mlp = MLP(layers, dropout)
        start_time = time.time()
        losses, train_accuracies, test_accuracies = mlp.train(train_x, train_y, test_x, test_y, alpha, batch, epochs)
        print("Took " + str(time.time() - start_time) + " seconds to train")
        plt.subplot(2, 1, 1)
        plt.plot(losses)
        plt.subplot(2, 1, 2)
        plt.plot(train_accuracies, ls="none", marker='o')
        plt.plot(test_accuracies, ls="none", marker='o')
        plt.show()

        with open('trained_conv_cifar.pkl', 'wb') as file:
            mlp_data = ([w.get_value() for w in mlp.weights], [b.get_value() for b in mlp.biases])
            pickle.dump((mlp_data, train_accuracies, test_accuracies, losses), file)
    else:
        with open('trained_conv_cifar.pkl', 'rb') as file:
            pre_trained, train_accuracies, test_accuracies, losses = saved_data = pickle.load(file)
            mlp = MLP(layers, dropout, pre_trained)

        plt.subplot(2, 1, 1)
        plt.plot(losses)
        plt.subplot(2, 1, 2)
        plt.plot(train_accuracies, ls="none", marker='o')
        plt.plot(test_accuracies, ls="none", marker='o')
        plt.show()
        print("Train accuracy: " + str(calc_accuracy(mlp, train_x, train_y)))
        print("Test accuracy: " + str(calc_accuracy(mlp, test_x, test_y)))


    # Show first layer weights
    # n_rows = 10
    # n_cols = int(n_hidden / n_rows)
    # gspec = matplotlib.gridspec.GridSpec(n_rows, n_cols)
    # gspec.update(wspace=0.05, hspace=0.05)
    # # f, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    # for row in range(n_rows):
    #     for col in range(n_cols):
    #         weight_idx = row*n_cols + col
    #         ax = plt.subplot(gspec[weight_idx])
    #         ax.imshow(mlp.weights[0].get_value(borrow=True)[weight_idx,:].reshape((32,32,3), order='F'))#((28,28)), cmap='gray')
    #         ax.axis('off')
    # plt.show()

    output = mlp.predict(test_x)
    predictions = np.argmax(output, axis=0)

    name_map = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    # show_predictions(test_x, predictions, name_map, 80)

