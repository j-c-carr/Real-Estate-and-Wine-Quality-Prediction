import numpy as np

class StochasticGradientDescent:
    """
    Stochastic gradient descent with momentum. Reduces to standard gradient descent when :batch_size: is equal to the
    number of samples in the training set. The parameter :beta: controls the momentum. When :beta: is equal to zero,
    the optimizer runs normal stochastic gradient descent. The StochasticGradientDescent class was modified from the 
    GradientDescent class defined in the gradient descent tutorial:
    https://github.com/rabbanyk/comp551-notebooks/blob/master/GradientDescent.ipynb
    """

    def __init__(self, learning_rate=0.1, max_iters=1e4, epsilon=1e-8, batch_size=1, record_history=False, verbose=True,
                 beta=0):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.verbose = verbose
        self.batch_size = batch_size
        self.beta = beta    # momentum
        self.record_history = record_history
        self.w_history = None

    def run(self, gradient_fn, X, y, w):
        assert self.batch_size <= X.shape[0], f'Error, batch size must be smaller than {X.shape[0]}'
        ix_list = [i for i in range(X.shape[0])]    # possible indices for each mini batch

        if self.record_history:
            self.w_history = np.empty((int(self.max_iters), *w.shape))

        grad = np.inf
        prev_delta_w = np.zeros(w.shape)
        t = 0
        while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:

            # update weights according to the equation in slide 30 of:
            # https://www.cs.mcgill.ca/~isabeau/COMP551/F23/slides/5-gradientdescent.pdf
            batch = np.random.choice(ix_list, size=self.batch_size, replace=False)

            grad = gradient_fn(X[batch], y[batch], w)

            delta_w = self.beta * prev_delta_w + (1-self.beta) * grad
            prev_delta_w = delta_w

            w = w - self.learning_rate * delta_w

            if self.verbose and (t % 100 == 0):
                print(f'gradient norm at step {t}: {np.linalg.norm(grad)}')

            if self.record_history:
                self.w_history[t] = w
            t += 1
        return w


class Adam:
    """
    Implements the Adam gradient descent method according to the weight update in slide 35 of:
    https://www.cs.mcgill.ca/~isabeau/COMP551/F23/slides/5-gradientdescent.pdf
    """

    def __init__(self, learning_rate=0.1, max_iters=1e5, epsilon=1e-8, batch_size=1, record_history=False, verbose=True,
                 beta_1=0.9, beta_2=0.9):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.verbose = verbose
        self.batch_size = batch_size
        self.beta_1 = beta_1    # used for moving average of the first moment
        self.beta_2 = beta_2    # used for moving average of the second moment

        self.record_history = record_history
        self.w_history = None

    def run(self, gradient_fn, X, y, w):
        assert self.batch_size <= X.shape[0], f'Error, batch size must be smaller than {X.shape[0]}'
        ix_list = [i for i in range(X.shape[0])]    # possible indices for each mini batch

        if self.record_history:
            self.w_history = np.empty((int(self.max_iters), *w.shape))

        grad = np.inf
        prev_M = np.zeros(w.shape)
        prev_S = np.zeros(w.shape)
        beta_1_t = 1
        beta_2_t = 1

        t = 0
        while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:

            batch = np.random.choice(ix_list, size=self.batch_size, replace=False)

            grad = gradient_fn(X[batch], y[batch], w)

            # Compute weighted moving average of first and second moment of the cost gradient
            M = self.beta_1 * prev_M + (1-self.beta_1) * grad
            S = self.beta_2 * prev_S + (1-self.beta_2) * grad**2
            prev_M = M
            prev_S = S

            beta_1_t *= self.beta_1
            beta_2_t *= self.beta_2
            M_hat = M / (1 - beta_1_t)
            S_hat = S / (1 - beta_2_t)

            w = w - (self.learning_rate * M_hat / np.sqrt(S_hat + self.epsilon))

            if self.verbose and (t % 100 == 0):
                print(f'gradient norm at step {t}: {np.linalg.norm(grad)}')

            if self.record_history:
                self.w_history[t] = w

            t += 1
        return w
