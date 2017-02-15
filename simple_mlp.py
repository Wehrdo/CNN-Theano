import matplotlib
import numpy as np
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import function as Tfunc
from theano import shared
import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.colors
import math
import time
import pickle
import itertools

MARGIN = 1

class MLP:
    def __init__(self, layer_sizes, dropout_rate, trained_model=None):
        n_layers = len(layer_sizes) - 1

        if trained_model is None:
            self.weights = []
            self.biases = []
            # Initialize random values for weight and biase matrices
            for i in range(1, n_layers + 1):
                w = np.random.standard_normal((layer_sizes[i], layer_sizes[i-1])) * math.sqrt(2 / layer_sizes[i-1])
                b = np.random.standard_normal((layer_sizes[i], 1)) * math.sqrt(2 / layer_sizes[i-1])
                self.weights.append(w)
                self.biases.append(b)
        else:
            self.weights = trained_model[0]
            self.biases = trained_model[1]

        #### Theano definitions ####

        # Turn weights and biases into shared variables to be used in training and predicting
        self.weights = [shared(w, f"w{i}") for i, w in enumerate(self.weights)]
        self.biases = [shared(b, f"b{i}", broadcastable=(False,True)) for i, b in enumerate(self.biases)]

        # Variable for y value (correct values when training)
        y = T.dmatrix('y')

        # List of matrices for each layer's activations, plus one for input
        activations = T.dmatrices(n_layers + 1)
        # Dropout random number matrices
        dropout_masks = []
        np.random.seed(1)
        np_rng = np.random.RandomState()
        for i in range(1, n_layers):
            srng = RandomStreams(np_rng.randint(1000000))
            rv = srng.uniform((layer_sizes[i], y.shape[1]))
            dropout_masks.append(rv)
        for l_i in range(n_layers):
            # Define layer function: max(0, w*x + b)
            if l_i == n_layers - 1: # Output layer
                activations[l_i + 1] = self.biases[l_i] + T.dot(self.weights[l_i], activations[l_i])
            # elif l_i == 0:
            #     activations[l_i + 1] = T.nnet.relu(self.biases[l_i] + T.dot(self.weights[l_i], activations[l_i]))
            else:
                pre_mask = T.nnet.relu(self.biases[l_i] + T.dot(self.weights[l_i], activations[l_i]))
                dropout_mask = (dropout_masks[l_i] < dropout_rate) / dropout_rate
                activations[l_i + 1] = pre_mask * dropout_mask


        ## Hinge loss
        # Indices of the correct classes
        correct_classes = T.argmax(y, axis=0, keepdims=True)
        # Actual values predicted for the correct classes
        correct_vals = activations[-1][correct_classes, T.arange(y.shape[1])]
        # margin: max(0, predicted - correct_value + 1)
        margin_mat = T.maximum(0, activations[-1] - T.repeat(correct_vals, repeats=y.shape[0], axis=0) + 1)
        # Loss for each training sample
        individual_losses = T.sum(margin_mat, axis=0) - margin_mat[correct_classes, T.arange(y.shape[1])]
        tot_loss = T.sum(individual_losses)
        # Average loss
        loss = tot_loss / y.shape[1]

        # List of matrices for each layer's activations, plus one for input
        pred_activations = T.dmatrices(n_layers + 1)
        for l_i in range(n_layers):
            # Define layer function: max(0, w*x + b)
            if l_i == n_layers - 1: # Output layer
                pred_activations[l_i + 1] = self.biases[l_i] + T.dot(self.weights[l_i], pred_activations[l_i])
            else:
                pred_activations[l_i + 1] = T.nnet.relu(self.biases[l_i] + T.dot(self.weights[l_i], pred_activations[l_i]))

        # Create Theano function for predicting
        self.predict = Tfunc([pred_activations[0]], pred_activations[-1])

        # List of expressions for derivatives: d_w1, d_w2, ... d_b1, d_b2,...
        derivatives = T.grad(loss, self.weights + self.biases)
        # Learning rate
        rate = T.dscalar('r')
        # How to update weights and biases when training
        B1 = 0.9
        B2 = 0.999
        eps = 1e-8
        decayed_B1 = shared(B1, 'B1')
        decayed_B2 = shared(B2, 'B2')
        update_rules = []
        for i, param in enumerate(itertools.chain(self.weights, self.biases)):
            param_dims = param.get_value(borrow=True).shape
            moment1 = shared(np.zeros(param_dims), broadcastable=param.broadcastable)
            moment2 = shared(np.zeros(param_dims), broadcastable=param.broadcastable)
            # computation
            gradient = derivatives[i]
            new_moment1 = (B1 * moment1) + ((1 - B1) * gradient)
            new_moment2 = (B2 * moment2) + ((1 - B2) * gradient * gradient)
            moment1_est = new_moment1 / (1 - decayed_B1)
            moment2_est = new_moment2 / (1 - decayed_B2)
            param_update = param - rate*(moment1_est / (T.sqrt(moment2_est) + eps))
            update_rules.append((param, param_update))
            update_rules.append((moment1, new_moment1))
            update_rules.append((moment2, new_moment2))
        update_rules.append((decayed_B1, B1 * decayed_B1))
        update_rules.append((decayed_B2, B2 * decayed_B2))


        # the_updates = [(var, var - rate*d_var) for var, d_var in zip(self.weights + self.biases, derivatives)]
        # Function for actually executing training
        self.update_step = Tfunc([activations[0], y, rate], loss, updates=update_rules)

    def train(self, x, y, test_x, test_y, rate=0.002, batch_size=200, epochs=30):
        losses = []
        train_accuracies = []
        test_accuracies = []
        for epoch in range(epochs):
            n_iters = int(x.shape[1] / batch_size)
            for iter in range(n_iters):
                selection = np.random.randint(0, x.shape[1], batch_size)
                # loss = self.update_weights(x[:,selection], y[:,selection], rate)
                loss = self.update_step(x[:,selection], y[:,selection], rate)
                losses.append(loss)
            train_accuracies.append(calc_accuracy(self, x, y))
            test_accuracies.append(calc_accuracy(self, test_x, test_y))
            print(f"Epoch {epoch}, train: {train_accuracies[-1]}, test: {test_accuracies[-1]}")
        return losses, train_accuracies, test_accuracies

    def update_weights(self, x, y, rate):
        N = x.shape[1]

        forward_ps = []
        forward_zs = []
        forward_as = []

        a = x
        forward_as.append(x)
        for w, b in zip(self.weights, self.biases):
            p = np.dot(w, a)
            z = p + b
            a = self.activation_func(z)
            forward_ps.append(p)
            forward_zs.append(z)
            forward_as.append(a)

        a_o = forward_as.pop()
        loss = self.compute_loss(y, a_o)
        d_loss = self.loss_gradient(y, a_o)

        d_bs = []
        d_ws = []
        d_a = d_loss
        for p, z, a, w in zip(*map(reversed, [forward_ps, forward_zs, forward_as, self.weights])):
            d_z = d_a * self.activation_gradient(z)
            d_b = np.sum(d_z, axis=1) / N
            d_p = d_z
            d_w = np.dot(d_p, a.T) / N
            d_a = np.dot(w.T, d_p)
            d_bs.append(d_b)
            d_ws.append(d_w)


        # Make in order of layer 0, layer 1, ...
        d_ws.reverse()
        d_bs.reverse()
        for w, d_w, b, d_b in zip(self.weights, d_ws, self.biases, d_bs):
            w -= rate * d_w
            b -= rate * d_b.reshape(-1,1)

        return loss

    def compute_loss(self, expected, actual):
        # return 0.5 * np.sum(np.square(actual - expected))
        correct_classes = np.nonzero(expected.T)[1]
        correct_vals = actual[correct_classes,np.arange(actual.shape[1])]
        margin_mat = np.maximum(0, actual - np.repeat(correct_vals.reshape(1,-1), repeats=expected.shape[0], axis=0) + 1)
        individual_losses = np.sum(np.ma.array(margin_mat, mask=expected), axis=0)
        loss = np.sum(individual_losses)
        N = actual.shape[1]
        return loss / N

    def loss_gradient(self, expected, actual):
        # return actual - expected
        N = actual.shape[1]
        correct_classes = np.nonzero(expected.T)[1]
        correct_vals = actual[correct_classes,np.arange(N)]
        margin_mat = np.maximum(0, actual - np.repeat(correct_vals.reshape(1,-1), repeats=expected.shape[0], axis=0) + 1)
        grad_mat = (margin_mat > 0).astype(float)
        grad_mat[correct_classes, np.arange(N)] = -np.ma.array(grad_mat, mask=expected).sum(axis=0)
        gradients = grad_mat.sum(axis = 1) / N
        return gradients.reshape(-1,1)
        # return grad_mat.sum() / N

    def activation_func(self, x):
        # return np.tanh(x)
        return np.maximum(x, 0)

    def activation_gradient(self, x):
        return self.relu_gradient(x)

    def relu_gradient(self, x):
        return (x > 0).astype(float)

    def tanh_gradient(self, x):
        e_2x = np.exp(2*x) # e^2x
        numerator = 4 * e_2x
        denominator = (e_2x + 1)**2
        return numerator / denominator

def preprocess(data, params=None):
    if params == None:
        mean = np.mean(data, axis=0)
        data -= mean
        stdev = np.std(data, axis=0)
        # data /= stdev
        return (mean, stdev)
    else:
        data -= params[0]
        # data /= params[1]
        return params

def calc_accuracy(mlp, data_x, data_y):
    output = mlp.predict(data_x)
    predictions = np.argmax(output, axis=0)
    accuracy = np.sum(np.argmax(data_y, axis=0) == predictions) / data_y.shape[1]
    return accuracy

def transform_labels(labels, n_classes=None):
    if n_classes is None:
        n_classes = len(np.unique(labels))
    y = np.zeros((n_classes, labels.shape[0]))
    for c in range(n_classes):
        y[c, np.where(labels == c)[0]] = 1
    return y

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

if __name__ == '__main__':
    # with open('datasets/mnist.pkl3', 'rb') as data_f:
    with open('datasets/cifar-10-python/combined_data.pkl3', 'rb') as data_f:
        train_set, test_set, validation_set = pickle.load(data_f)
    # with gzip.open('datasets/mnist.pkl3.gz', 'wb') as data_f:
    #     pickle.dump((train_set, test_set, validation_set), data_f)

    amt = 50000
    train_data = train_set[0][0:amt]
    train_labels = train_set[1][0:amt]

    transform = preprocess(train_data)

    train_x = train_data.T
    # n_classes = len(np.unique(train_labels))
    train_y = transform_labels(train_labels, 10)

    test_y = transform_labels(test_set[1], 10)
    test_data = test_set[0]
    preprocess(test_data, transform)
    test_x = test_data.T
    # adam, RMSprop
    train_scratch = True
    n_hidden = 100
    layer_sizes = [train_x.shape[0], 100,  10]
    dropout = 0.7
    batch = 200
    epochs = 15
    alpha = 0.001

    if train_scratch:
        mlp = MLP(layer_sizes, dropout)

        start_time = time.time()
        losses, train_accuracies, test_accuracies = mlp.train(train_x, train_y, test_x, test_y, alpha, batch, epochs)
        print("Took " + str(time.time() - start_time) + " seconds to train")
        plt.subplot(2, 1, 1)
        plt.plot(losses)
        plt.subplot(2, 1, 2)
        plt.plot(train_accuracies, ls="none", marker='o')
        plt.plot(test_accuracies, ls="none", marker='o')
        plt.show()

        with open('trained_model_cifar.pkl', 'wb') as file:
            mlp_data = ([w.get_value() for w in mlp.weights], [b.get_value() for b in mlp.biases])
            pickle.dump((mlp_data, train_accuracies, test_accuracies, losses), file)
    else:
        with open('trained_model_cifar.pkl', 'rb') as file:
            pre_trained, train_accuracies, test_accuracies, losses = saved_data = pickle.load(file)
            mlp = MLP(layer_sizes, dropout, pre_trained)

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

