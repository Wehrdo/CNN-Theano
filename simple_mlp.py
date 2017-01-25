import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec
import math
import pickle

MARGIN = 1

class MLP:
    def __init__(self, n_d, n_h, n_c):
        self.weights = []
        self.biases = []
        w1 = np.random.standard_normal((n_h, n_d)) * math.sqrt(2/n_d)
        b1 = np.random.standard_normal((n_h, 1)) * math.sqrt(2/n_d)
        self.weights.append(w1)
        self.biases.append(b1)

        w2 = np.random.standard_normal((n_c, n_h)) * math.sqrt(2/n_h)
        b2 = np.random.standard_normal((n_c, 1)) * math.sqrt(2/n_h)
        self.weights.append(w2)
        self.biases.append(b2)

    def compute(self, x):
        a1 = self.activation_func(self.biases[0] + np.dot(self.weights[0], x))
        a2 = self.activation_func(self.biases[1] + np.dot(self.weights[1], a1))
        return a2

    def train(self, x, y):
        losses = []
        batch_size = 10
        for epoch in range(2):
            n_iters = int(x.shape[1] / batch_size)
            for iter in range(n_iters):
                selection = np.random.randint(0, x.shape[1], batch_size)
                loss = self.update_weights(x[:,selection], y[:,selection], 0.01 / (epoch + 1))
                losses.append(loss)
        return losses

    def update_weights(self, x, y, rate):
        N = x.shape[1]

        p1 = np.dot(self.weights[0], x)
        z1 = p1 + self.biases[0]
        a1 = self.activation_func(z1)

        p2 = np.dot(self.weights[1], a1)
        z2 = p2 + self.biases[1]
        a2 = self.activation_func(z2)

        loss = self.compute_loss(y, a2)
        d_loss = self.loss_gradient(y, a2)

        d_z2 = d_loss * self.activation_gradient(z2)
        d_b2 = np.sum(d_z2, axis=1) / N # * 1
        d_p2 = d_z2 # * 1
        d_w2 = np.dot(d_p2, a1.T)
        d_a1 = np.dot(self.weights[1].T, d_p2)

        d_z1 = d_a1 * self.activation_gradient(z1)
        d_b1 = np.sum(d_z1, axis=1) / N
        d_p1 = d_z1
        d_w1 = np.dot(d_p1, x.T)

        self.weights[0] -= rate * d_w1
        self.weights[1] -= rate * d_w2
        self.biases[0] -= rate * d_b1.reshape(-1,1)
        self.biases[1] -= rate * d_b2.reshape(-1,1)
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
        N = x.shape[1]
        # return np.sum(x > 0, axis=1).astype(float) / N
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

    n_hidden = 100
    mlp = MLP(train_x.shape[0], n_hidden, 10)

    losses = mlp.train(train_x, train_y)


    n_rows = 10
    n_cols = int(n_hidden / n_rows)
    gspec = matplotlib.gridspec.GridSpec(n_rows, n_cols)
    gspec.update(wspace=0.05, hspace=0.05)
    # f, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    for row in range(n_rows):
        for col in range(n_cols):
            weight_idx = row*n_cols + col
            ax = plt.subplot(gspec[weight_idx])
            ax.imshow(mlp.weights[0][weight_idx,:].reshape((28,28)), cmap='gray')
            ax.axis('off')
    plt.show()


    train_results = mlp.compute(train_x)
    predictions = np.argmax(train_results, axis=0)

    accuracy = np.sum(np.argmax(train_y, axis=0) == predictions) / train_y.shape[1]
    print("Train accuracy: " + str(accuracy))

    test_y = transform_labels(test_set[1], 10)
    test_data = test_set[0]
    preprocess(test_data, transform)
    test_x = test_data.T
    test_accuracy = np.sum(np.argmax(test_y, axis=0) == np.argmax(mlp.compute(test_x), axis=0)) / test_y.shape[1]
    print("Test accuracy: " + str(test_accuracy))
    
    plt.plot(losses)
    plt.show()

    # train_data = []
    # train_size = 100
    # for i in range(train_size):
    #     x, y = np.random.randint(0, 2, 2)
    #     xor = x ^ y
    #     train_data.append((np.array([x,y]).reshape((2,1)), (np.array([xor]))))
    # # for x_raw in range(2):
    # #     for y_raw in range(2):
    # #         xor = x_raw ^ y_raw
    # #         x, y = (x_raw * 2 - 1, y_raw * 2 - 1)
    # #         xor = xor * 2 - 1
    # #         train_data.append((np.array([x, y]).reshape((2, 1)), (np.array([xor]))))
    #
    #
    # normalization = preprocess(train_data)
    #
    # mlp.train(train_data)
    #
    # for sample in [(0 ,0), (0, 1), (1, 0), (1, 1)]:
    #     output = mlp.compute(np.array(sample).reshape((2,1)))
    #     print(sample, output)
    #
    # zs = []
    # steps = 100
    # for x in np.linspace(-0.5, 1.5, steps):
    #     for y in reversed(np.linspace(-0.5, 1.5, steps)):
    #         zs.append(mlp.compute(np.array([x,y]).reshape((2,1))))
    # values = np.array(zs).reshape((steps, steps))
    # plt.imshow(values, extent=(-0.5, 1.5, -0.5, 1.5), interpolation='nearest', cmap=cm.gist_heat)
    # plt.show()
