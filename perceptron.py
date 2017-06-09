"""Perceptron learing demostration."""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Data:
    def __init__(self, fname):
        self.fname = fname
        data = scipy.io.loadmat(fname)
        # The num_neg_examples x 2 matrix for the examples with target 0.
        neg = data['neg_examples_nobias']
        # The num_pos_examples x 2 matrix for the examples with target 1.
        pos = data['pos_examples_nobias']
        # Add a column of ones to the examples in order to allow us to learn bias parameters.
        self.neg_examples = np.concatenate((neg, np.ones((len(neg), 1))), 1)
        self.pos_examples = np.concatenate((pos, np.ones((len(pos), 1))), 1)
        # Initial model
        self.w_init = data['w_init']

    def save(self, fname=None):
        if fname is None:
            fname = self.fname
        data = {}
        data['neg_examples_nobias'] = self.neg_examples[:, :-1]
        data['pos_examples_nobias'] = self.pos_examples[:, :-1]
        data['w_init'] = self.w_init
        scipy.io.savemat(fname, data)


# -----------------------------------------------------------------------------
# Learn perceptron

def learn_perceptron(data, learning_rate=0.01):
    """Learns the weights of a perceptron for a 2-dimensional dataset."""

    ITER_MAX = 120          # number of learning iterations
    epsilon = 0.001         # classification margin
    w = data.w_init.copy()  # initial weight vector
    w_history = np.zeros([3, ITER_MAX])
    err_neg_history = np.zeros([len(data.neg_examples), ITER_MAX], dtype = np.int)
    err_pos_history = np.zeros([len(data.pos_examples), ITER_MAX], dtype = np.int)
    err_num_history = np.zeros([ITER_MAX], dtype = np.int)

    # Learning cycle
    for iter in range(ITER_MAX):
        # Save current model to history
        w_history[:, iter] = w[:, 0]
        # Find the data points that the perceptron has incorrectly classified.
        # and record the number of errors it makes.
        err0, err1 = eval_perceptron(data.neg_examples, data.pos_examples, w)
        err_neg_history[:, iter] = err0
        err_pos_history[:, iter] = err1
        err_num_history[iter] = sum(err0) + sum(err1)
        update_weights(data.neg_examples, data.pos_examples, w, learning_rate, epsilon)

    return data, w_history, err_neg_history, err_pos_history, err_num_history


def update_weights(neg_examples, pos_examples, w, learning_rate, epsilon):
    """Updates the weights of the perceptron for incorrectly classified points
    using the perceptron update algorithm. This function makes one sweep
    over the dataset."""

    for i in range(len(neg_examples)):
        input = np.array(neg_examples[i], ndmin=2)
        output = np.dot(input, w)
        if output >= -epsilon:
            w -= input.T * learning_rate

    for i in range(len(pos_examples)):
        input = np.array(pos_examples[i], ndmin=2)
        output = np.dot(input, w)
        if output < epsilon:
            w += input.T * learning_rate


def eval_perceptron(neg_examples, pos_examples, w):
    """Evaluates the perceptron using a given weight vector. Here, evaluation
    refers to finding the data points that the perceptron incorrectly classifies."""

    err0 = np.array([np.dot(np.array(neg_examples[i], ndmin=2), w) >= 0
                for i in range(len(neg_examples))], dtype=np.int).ravel()

    err1 = np.array([np.dot(np.array(pos_examples[i], ndmin=2), w)  < 0
                for i in range(len(pos_examples))], dtype=np.int).ravel()
    
    return err0, err1


# -----------------------------------------------------------------------------
# Plot perceptron

def plot_perceptrons(p1, p2, p3, p4):
    """Plots information about a perceptron classifier on a 2-dimensional dataset.
    
    The left plot shows the dataset and the classification boundary given by
    the weights of the perceptron. The negative examples are shown as circles
    while the positive examples are shown as squares. If an example is colored
    green then it means that the example has been correctly classified by the
    provided weights. If it is colored red then it has been incorrectly classified.
    The right plot shows the number of mistakes the perceptron algorithm has
    made in each iteration so far. """
    
    fig = plt.figure(figsize=(10,12))
    fig.set_tight_layout(True)
    axes = {}
    plot1 = setup_perceptron_plot(p1, 1, axes)
    plot2 = setup_perceptron_plot(p2, 2, axes)
    plot3 = setup_perceptron_plot(p3, 3, axes)
    plot4 = setup_perceptron_plot(p4, 4, axes)

    ani = animation.FuncAnimation(fig, animate, fargs=(plot1, plot2, plot3, plot4),
        blit=True, interval=67, frames=len(p1[4]), repeat=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)
    ani.save('perceptrons.mp4', writer=writer)
    # plt.show()    


def setup_perceptron_plot(p, n, axes):
    # Unpack arguments
    data, w_history, err_neg_history, err_pos_history, err_num_history = p
    plot_pos = n * 2 - 1

    # Set up classifier plot
    ax = 'class' + str(n)
    axes[ax] = plt.subplot(4, 2, plot_pos)
    if n == 1:
        axes[ax].set_title("Classifier")
    axes[ax].set_xlim(-1, 1);
    axes[ax].set_ylim(-1, 1);
    # Data
    ln_neg_good, = axes[ax].plot([], [], "og", markersize=10)
    ln_pos_good, = axes[ax].plot([], [], "^g", markersize=10)
    ln_neg_bad,  = axes[ax].plot([], [], "or", markersize=10)
    ln_pos_bad,  = axes[ax].plot([], [], "^r", markersize=10)
    # Classifier
    ln_class, = axes[ax].plot([], [])

    # Set up error plot
    ax = 'err' + str(n + 1)
    axes[ax] = plt.subplot(4, 2, plot_pos+1)
    if n == 1:
        axes[ax].set_title("Number of Errors")
    axes[ax].set_xlabel("Iterations")
    axes[ax].set_ylabel("Number of errors")
    axes[ax].set_xlim(-1, len(err_num_history))
    axes[ax].set_ylim(0, len(data.neg_examples) + len(data.pos_examples) + 1)
    # Errors
    ln_err, = axes[ax].plot([], [])

    history = [w_history, err_neg_history, err_pos_history, err_num_history]
    lines = [ln_neg_good, ln_pos_good, ln_neg_bad, ln_pos_bad, ln_class, ln_err]

    return data, history, lines


def animate(i, plot1, plot2, plot3, plot4):
    lines = []
    lines.extend(update_plot(i, *plot1))
    lines.extend(update_plot(i, *plot2))
    lines.extend(update_plot(i, *plot3))
    lines.extend(update_plot(i, *plot4))
    return lines


def update_plot(i, data, history, lines):
    # Unpack arguments
    w_history, err_neg_history, err_pos_history, err_num_history = history
    ln_neg_good, ln_pos_good, ln_neg_bad, ln_pos_bad, ln_class, ln_err = lines

    # Update data
    neg_good_idx = np.nonzero(err_neg_history[:, i] == 0)[0]
    neg_bad_idx  = np.nonzero(err_neg_history[:, i])[0]
    pos_good_idx = np.nonzero(err_pos_history[:, i] == 0)[0]
    pos_bad_idx  = np.nonzero(err_pos_history[:, i])[0]

    neg = data.neg_examples
    pos = data.pos_examples

    if len(neg_good_idx):
        ln_neg_good.set_data(neg[neg_good_idx, 0], neg[neg_good_idx, 1])
    else:
        ln_neg_good.set_data([], [])

    if len(pos_good_idx):
        ln_pos_good.set_data(pos[pos_good_idx, 0], pos[pos_good_idx, 1])
    else:
        ln_pos_good.set_data([], [])

    if len(neg_bad_idx):
        ln_neg_bad.set_data(neg[neg_bad_idx, 0], neg[neg_bad_idx, 1])
    else:
        ln_neg_bad.set_data([], [])

    if len(pos_bad_idx):
        ln_pos_bad.set_data(pos[pos_bad_idx, 0], pos[pos_bad_idx, 1])
    else:
        ln_pos_bad.set_data([], [])

    # Update classifier
    w = w_history[:, i]
    ln_class.set_data([-5, 5], [(w[0] * 5 - w[2]) / w[1], (- w[0] * 5 - w[2]) / w[1]])

    # Update errors
    ln_err.set_data(range(i), err_num_history[:i])

    return lines


# -----------------------------------------------------------------------------
# Main

p1 = learn_perceptron(Data('data1'), 0.01)
p2 = learn_perceptron(Data('data2'), 0.02)
p3 = learn_perceptron(Data('data3'), 0.01)
p4 = learn_perceptron(Data('data4'), 0.005)

plot_perceptrons(p1, p2, p3, p4)
