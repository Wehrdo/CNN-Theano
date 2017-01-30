import matplotlib
import numpy as np
import theano
import theano.tensor as T
from theano import function as Tfunc
from theano import shared, typed_list
import matplotlib.pyplot as plt
import matplotlib.gridspec
import math
import time
import pickle

MARGIN = 1

class MLP:
    def __init__(self, layer_sizes, trained_model=None):
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

        # List of matrices for each layer's activations, plus one for input
        activations = T.dmatrices(n_layers + 1)
        for l_i in range(n_layers):
            # Define layer function: max(0, w*x + b)
            activations[l_i+1] = T.nnet.relu(self.biases[l_i] + T.dot(self.weights[l_i], activations[l_i]))

        ## Hinge loss
        # Variable for y value (correct values when training)
        y = T.dmatrix('y')
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

        # Create Theano function for predicting
        self.predict = Tfunc([activations[0]], activations[-1])

        # List of expressions for derivatives: d_w1, d_w2, ... d_b1, d_b2,...
        derivatives = T.grad(loss, self.weights + self.biases)
        # Learning rate
        rate = T.dscalar('r')
        # How to update weights and biases when training
        the_updates = [(var, var - rate*d_var) for var, d_var in zip(self.weights + self.biases, derivatives)]
        # Function for actually executing training
        self.update_step = Tfunc([activations[0], y, rate], loss, updates=the_updates)

    def train(self, x, y):
        losses = []
        batch_size = 200
        rate = 0.3
        for epoch in range(30):
            n_iters = int(x.shape[1] / batch_size)
            for iter in range(n_iters):
                selection = np.random.randint(0, x.shape[1], batch_size)
                # loss = self.update_weights(x[:,selection], y[:,selection], rate)
                loss = self.update_step(x[:,selection], y[:,selection], rate)
                losses.append(loss)
            rate *= 0.95
        return losses

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

def transform_labels(labels, n_classes=None):
    if n_classes is None:
        n_classes = len(np.unique(labels))
    y = np.zeros((n_classes, labels.shape[0]))
    for c in range(n_classes):
        y[c, np.where(labels == c)[0]] = 1
    return y


if __name__ == '__main__':
    with open('datasets/mnist.pkl3', 'rb') as data_f:
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

    # with open('trained_model.pkl', 'rb') as file:
    #     pre_trained = pickle.load(file)

    n_hidden = 100
    layer_sizes = [train_x.shape[0], n_hidden,  10]
    mlp = MLP(layer_sizes)

    start_time = time.time()
    losses = mlp.train(train_x, train_y)
    print("Took " + str(time.time() - start_time) + " seconds to train")
    # 50 seconds to train 100 hidden nodes with 7 epochs and batch size 10


    # with open('trained_model.pkl', 'wb') as file:
    #     pre_trained = pickle.dump((mlp.weights, mlp.biases), file)

    # Show first layer weights
    n_rows = 10
    n_cols = int(n_hidden / n_rows)
    gspec = matplotlib.gridspec.GridSpec(n_rows, n_cols)
    gspec.update(wspace=0.05, hspace=0.05)
    # f, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    for row in range(n_rows):
        for col in range(n_cols):
            weight_idx = row*n_cols + col
            ax = plt.subplot(gspec[weight_idx])
            ax.imshow(mlp.weights[0].get_value(borrow=True)[weight_idx,:].reshape((28,28)), cmap='gray')
            ax.axis('off')
    plt.show()


    train_results = mlp.predict(train_x)
    predictions = np.argmax(train_results, axis=0)

    accuracy = np.sum(np.argmax(train_y, axis=0) == predictions) / train_y.shape[1]
    print("Train accuracy: " + str(accuracy))

    test_y = transform_labels(test_set[1], 10)
    test_data = test_set[0]
    preprocess(test_data, transform)
    test_x = test_data.T
    test_accuracy = np.sum(np.argmax(test_y, axis=0) == np.argmax(mlp.predict(test_x), axis=0)) / test_y.shape[1]
    print("Test accuracy: " + str(test_accuracy))
    
    plt.plot(losses)
    plt.show()
