import numpy as np


class LinearRegression:
    """
    Linear regression

    Parameters
    __________
    lr : float, default = 0.01
        Learning rate for gradient descend
    penalty : {'l1', 'l2', 'None'}, default = 'l1'
        Regularization type
    C : float, default = 0.1
        Regularization term
    max_epoch : int, default = 500
        Epochs of gradient descent
    weights : {'random', 'zero'}
        Intialization of weights. Random for np.rand, zero for {0, 0, 0..}
    fit_intercept : bool
        Add an intercept term as first column of features array
    """
    def __init__(self, lr: float = 0.01, penalty: str = 'l1', C: float = 0.1, max_epoch: int = 500,
                 weights: str = 'random', fit_intercept: bool = True):
        self.lr = lr
        self.penalty = {
            'l1': penalty == 'l1',
            'l2': penalty == 'l2',
        }
        self.C = C
        assert weights in ['zero', 'random'], "weights parameter should be in {'zero', 'random'}"
        self.weights = weights
        self.fit_intercept = fit_intercept
        self.max_epoch = max_epoch
        self.fit_loss = []
        self.loss = None
        self.w = np.array([0])

    def _loss(self, y, y_pred):
        loss = np.sum((y - y_pred) ** 2) / len(y) + \
               self.C * np.sum(abs(self.w)) * self.penalty['l1'] + \
               self.C * np.sum(self.w ** 2) * self.penalty['l2']
        return loss

    def _loss_grad(self, X, y):
        temp = np.dot((y - self.predict(X)), X.T)
        loss_grad = - 2 / len(y) * np.sum(temp) + \
                    self.C * np.sign(self.w) * self.penalty['l1'] + \
                    self.C * self.w * self.penalty['l2']
        return loss_grad

    def fit(self, X, y, verbose: int = 0):
        """
        Fit regressor instance to X given observed y.
        Iterates through gradient descent

        Params
        ______
        X : array-like
            Features array
        y : array-like
            Labels array
        verbose : int, default = False
            Print details about convergence
        """
        if self.fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.hstack(intercept, X)

        if self.weights == 'zero':
            self.w = np.zeros((1, X.shape[1]))
        elif self.weights == 'random':
            self.w = np.random.rand(1, X.shape[1])
        else:
            raise ValueError('weights parameter should be in {"zeros", "random"}')

        if y.shape != (1, X.shape[0]):
            y = y.reshape(-1, 1)

        for epoch in range(self.max_epoch):
            y_pred = self.predict(X)
            epoch_loss = self._loss(y, y_pred)
            self.fit_loss.append(epoch_loss)
            self.w = self.w - self.lr * self._loss_grad(X, y)
            if (verbose != 0) and (len(y) % verbose == 0):
                print(f'Epoch: {epoch}    |    Epoch loss: {epoch_loss}')
        self.loss = self._loss(y, self.predict(X))

    def predict(self, X):
        """
        Predict y from X

        Params
        ------
        X : array-like
            Feature-matrix

        Returns
        -------
        pred - nx1 vector of predictions
        """
        pred = np.dot(X, self.w.T)
        return pred


# ________________________________________________________________________


class LogisticRegression:
    """
    Linear regression

    Parameters
    __________
    lr : float, default = 0.01
        Learning rate for gradient descend
    penalty : {'l1', 'l2', 'None'}, default = 'l1'
        Regularization type
    C : float, default = 0.1
        Regularization term
    max_epoch : int, default = 500
        Epochs of gradient descent
    weights : {'random', 'zero'}
        Intialization of weights. Random for np.rand, zero for {0, 0, 0..}
    fit_intercept : bool
        Add an intercept term as first column of features array
    """
    def __init__(self, learning_rate=0.01, C=1, penalty='l1', init_type='random',
                 max_iter=500, convergence=1e-4):
        self.learning_rate = learning_rate
        self.C = C
        self.penalty = {'l1': int(penalty == 'l1'),
                        'l2': int(penalty == 'l2')}
        self.init_type = init_type
        self.max_iter = max_iter
        self.convergence = convergence
        self.epoch_loss_history = []

    def sigmoid(self, X, y):
        latent = (X @ self.w.T + self.bias)
        z = y * latent
        return (1 / (1 + np.exp(-z)))

    def logistic_loss(self, X, y):
        loss = (
                np.log(1 / self.sigmoid(X, y)).sum() / len(y) +  # logloss
                (self.C / 2 * (self.w @ self.w.T)) * self.penalty['l2'] +  # pen=l2
                (self.C * self.w.sum()) * self.penalty['l1']  # pen = l1
        )
        return loss

    def logistic_grad(self, X, y):
        loss_grad_w = (y.reshape(-1, 1) * X * (1 - self.sigmoid(X, y)).reshape(-1, 1)).sum(axis=0) / len(y)
        l2_grad_w = self.C * self.w * self.penalty['l2']
        l1_grad_w = (self.C * np.sign(self.w)) * self.penalty['l1']
        grad_w = loss_grad_w + l2_grad_w + l1_grad_w
        grad_b = (
                (y.reshape(-1, 1) * (1 - self.sigmoid(X, y))).sum(axis=0) / len(y)
        )
        self.w = self.w + self.learning_rate * grad_w
        self.bias = self.bias + self.learning_rate * grad_b

    def fit(self, X, y, verbose=False):
        """
        Fit classifier instance to X given observed y.
        Iterates through gradient descent

        Params
        ______
        X : array-like
            Features array
        y : array-like
            Labels array
        verbose : int, default = False
            Print details about convergence
        """
        if self.init_type == 'zeros':
            self.w = np.zeros(X.shape[1])
            self.bias = np.array([0])
        elif self.init_type == 'random':
            self.w = np.random.rand(X.shape[1])
            self.bias = np.random.rand(1)
        else:
            raise ValueError('init_type parameter should be in ["zeros", "random"]')

        for epoch in range(self.max_iter):
            prev_w = self.w
            epoch_loss = self.logistic_loss(X, y)
            self.epoch_loss_history.append(epoch_loss)
            self.logistic_grad(X, y)
            w_dif = np.linalg.norm(self.w - prev_w)
            if w_dif < self.convergence:
                print(f'Convergence achieved on epoch {epoch}')
                return

            if (verbose != 0) and (len(y) % verbose == 0):
                print(f'Epoch: {epoch}    |    Epoch loss: {epoch_loss}')
        print(f'Convergence is not achieved in {self.max_iter} epochs. Best criterion: {self.epoch_loss_history[-1]}')

    def predict(self, X, threshold: float = 0.5):
        """
        Predict y from X

        Params
        ------
        X : array-like
            Feature-matrix
        threshold : float
            Probability threshold for class prediction

        Returns
        -------
        pred - nx1 vector of predictions
        """
        pred = int(self.predict_proba(X) > threshold)
        return pred

    def predict_proba(self, X):
        """
        Predict probability of y from X

        Params
        ------
        X : array-like
            Feature-matrix

        Returns
        -------
        pred - nx1 vector of predictions
        """
        latent = (X @ self.w.T + self.bias)
        pred = (1 / (1 + np.exp(-latent)))
        return pred
