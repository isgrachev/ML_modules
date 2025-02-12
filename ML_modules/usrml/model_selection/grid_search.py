from itertools import product
import numpy as np


# define permutations from input parameters
# define grid_search (no c-v for now, maybe later)


class GridSearch:
    """
    Implement grid search for specified estimator over specified parameter grid

    Params:
    _______
    estimator : object
        Estimator. Has to have a .fit(X, y) built-in method.
    param_grid : dict
        Parameter grid with parameter names as keys, value range as values
    scoring : object
        Scoring function to compare estimators. Has to have (y_true, y_pred) parameters as inputs
    refit : bool, default = True
        Refit best estimator to input estimator instance
    verbose : bool, default = False
        Print the score of each parameter set

    Arguments:
    __________
    best_estimator, best_params, best_score, scores
        Self-explanatory. Scores - dict of scores (values) for all sets of parameters (keys)
    """
    def __init__(self, estimator, param_grid, scoring, refit : bool = True, verbose=False):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.refit = refit
        self.verbose = verbose
        self.best_estimator = None
        self.best_params = None
        self.best_score = np.inf
        self.scores = {}

    def _permutations(self, param_grid):
        """
        Get all combinations of parameters

        Params:
        _______
        param_grid: dict
            Parameter ranges

        Returns:
        ________
        List of parameter cominations
        """
        param_grid = [list(param) for param in param_grid.values()]
        res = [item for item in product(*param_grid)]
        return res

    def fit(self, X, y):
        """
        Fit estimators with parameters from the grid to a dataset

        Params:
        _______
        X : array-like
            Features
        y : array-like
            Labels
        """
        permutations = self._permutations(self.param_grid)
        for param_set in permutations:
            est = self.estimator(*param_set)
            est.fit(X, y)
            pred = est.predict(X)
            score = self.scoring(y, pred)
            self.scores[param_set] = score
            if score < self.best_score:
                self.best_score = score
                self.best_estimator = est
                self.best_params = param_set
            if self.verbose:
                print(f'Score {score} with parameters {param_set}')
        if self.refit:
            self.estimator = self.best_estimator
