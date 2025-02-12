from ..tree import DecisionTreeRegressor, DecisionTreeClassifier
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class GradientBoosting:
    """
    Gradient boosting on decision trees.
    Regressor uses MSE as loss function.
    Classifier is suitable for binary and multiclass problems. Uses entropy as loss function.
    ____________________________________________________________________________
    :params
        type: {'regressor', 'classifier'}
        base_model_params: dict, default=None
            Dictionary of base tree parameters. Takes all parameters of
            sklearn.tree.DecisionTreeRegressor
        n_estimators: int, default=10
            Amount of base models to be calculated (algorithm complexity)
        learning_rate: float, default=0.3
            Multiplier for base model. Additional multiplication with
            individual coefficients, therefore additional restriction on convergence
        subsample: float, default=0.3
            Subsample size for bootstrappig training subsample for each base model
        early_stopping_rounds: int, default=None
            Early stopping after specified iterations without improvement of
            quality on validation subsample
        plot: bool, default=False
            Plot train and validation loss history based on iterations to visualise
            convergence.

    attributes:
        models: list
            List of base models (Accessible trees)
        gammas: list
            List of coefficients of each model in prediction by linear combination
        history: dict
            Dictionary of loss function values over iterations
    """
    def __init__(
            self,
            type: str = 'classifier',
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.n_estimators = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate = learning_rate
        self.subsample = subsample

        self.early_stopping_rounds = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        if type == 'classifier':
            self.base_model_class = DecisionTreeClassifier
            self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
            self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        elif type == 'regressor':
            self.base_model_class = DecisionTreeRegressor
            self.loss_fn = lambda y, z: - ((y - z) ** 2)
            self.loss_derivative = lambda y, z: 2 * (y - z)
        else:
            raise ValueError('Parameter type should be in {"classifier", "regressor"}')

    def fit_new_base_model(self, x, y, predictions):
        # create new subsample. It would be better to not use train_test_split, but its only HW...
        # idx = np.arange(0, x.shape[0])
        train_idx = np.random.sample(range(x.shape[0]), (x.shape[0] * self.subsample))
        # train_idx, val_idx, _, _ = train_test_split(idx, idx, train_size=self.subsample)
        x_tr = x[train_idx]
        y_tr = y[train_idx]
        predictions_tr = predictions[train_idx]

        # do we need to take gradient of y by predictions here?
        # gradient of y by predictions on train sample
        # batch_loss = self.loss_fn(y_tr, predictions_tr)
        antigrad = -self.loss_derivative(y_tr, predictions_tr)

        # train the model to predict antigradients
        # temporarily update predictions by adding predictions for whole dataset
        base_model = self.base_model_class(**self.base_model_params)
        base_model.fit(x_tr, antigrad)
        new_predictions = predictions + base_model.predict(x)

        # search for gamma
        gamma = self.find_optimal_gamma(y, predictions, new_predictions)

        # debugging...
        # print(train_idx)
        # print('batch_loss', batch_loss)
        # print('antigrad', antigrad)
        # print('base model predictions', base_model.predict(x))
        # print('gamma', gamma)

        self.gammas.append(gamma)
        self.models.append(base_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        Fit classifier to x_train and y_train
        ___________________________________________
        :params
            x_train
                Features array (train set)
            y_train
                Targets array (train set)
            x_valid
                Features array (validation set)
            y_valid
                Targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        for _ in range(self.n_estimators):
            # fit base, update predictions and check for early stopping
            self.fit_new_base_model(x_train, y_train, train_predictions)
            new_train_predictions = self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_train)
            new_val_predictions = self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_valid)
            train_predictions = (train_predictions + new_train_predictions)
            valid_predictions = (valid_predictions + new_val_predictions)

            train_loss = self.loss_fn(y_train, train_predictions)
            valid_loss = self.loss_fn(y_valid, valid_predictions)
            self.history['train'].append(train_loss)
            self.history['validation'].append(valid_loss)

            if self.early_stopping_rounds is not None:
                # keep n last losses,
                # if all are bigger than n-1 loss, return
                benchmark_loss = self.validation_loss[0]
                self.validation_loss = np.append(self.validation_loss[1:],
                                                 valid_loss)
                if np.all(self.validation_loss < benchmark_loss):
                    return

        if self.plot:
            n = list(range(1, (self.n_estimators + 1)))
            plt.figure(figsize=(6, 6))
            plt.title('Ensemble loss based on n_estimators')
            plt.plot(n,
                     self.history['train'],
                     label='Train loss')
            plt.plot(n,
                     self.history['validation'],
                     label='Validation loss')
            plt.xlabel('n_estimators')
            plt.ylabel('Loss')
            plt.legend(loc='upper right')
            plt.show()

    def predict_proba(self, x):
        """
        Predict probabilities (binary problem only)
        ___________________________________________
        :params
            x: numpy.array
                Features array

        :returns
            numpy.array((n, 2)) with probabilities, where n is amout of observations,
            0 column has P(y==0), 1 column has P(y==1)
        """
        predictions = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            predictions = predictions + self.learning_rate * gamma * model.predict(x)

        probabilities = np.zeros((x.shape[0], 2))
        probabilities[:, 1] = self.sigmoid(predictions)
        probabilities[:, 0] = 1 - probabilities[:, 1]
        return probabilities

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=50)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    # @property
    # def feature_importances_(self):
    #     """
    #     Returns numpy array with element-wise sum of feature importances retrieved
    #     from each base algorithm scaled by gamma
    #     """
    #     feature_importances = np.zeros(self.models[1].feature_importances_.shape)
    #     for model, gamma in zip(self.models, self.gammas):
    #         # weigh by gamma and learning rate?
    #         feature_importances = feature_importances + gamma * model.feature_importances_
    #     return feature_importances

