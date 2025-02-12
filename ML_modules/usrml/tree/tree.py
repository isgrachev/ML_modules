import numpy as np
from collections import Counter, defaultdict


class DecisionTreeClassifier:
    """
    Decision tree classifier
    ____________________________________________________________________________
    :params
        feature_types: list
            List o feature types ('num' or 'categorical'). Determines encoding
        max_depth: int/float, default=+inf
            Maximum depth of the tree.
        min_samples_split: int, default=2
            Minimum samples requirement for a parent node to be split
        min_samples_leaf: int, default=1
            Minimum samples requirement in a child node to be considered
    """
    def __init__(self, feature_types, max_depth=np.inf, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")
        assert max_depth > 0
        assert min_samples_split > 1
        assert min_samples_leaf > 0

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _find_best_split(self, feature_vector, target_vector, min_samples_leaf=0):
        """
        Implementation of impurity-optimal split search by a single feature
        __________________________________________________________________________________________________
        :params
            feature_vector: numerical feature
            target_vector: target values
            min_samples_leaf: minimum samples requirement in a child node to be considered

        :returns
            thresholds: vector with all possible thresholds by which objects could be divided into
            two different subsamples, or subtree. It should be sorted in ascending order;
            ginis: a vector with the values of the Gini criterion for each of the thresholds respectively;
            threshold_best: optimal threshold;
            gini_best: Gini for the optimal threshold.
        """
        # check value adequacy:
        # length of feature and target vectors
        if len(feature_vector) != len(target_vector):
            raise ValueError('feature_vector and target_vector parameters must be of same lengths')
        unique = np.unique(feature_vector)
        # feature vector is not constant
        if len(unique) == 1:
            return {'thresholds': None,
                    'ginis': None,
                    'threshold': None,
                    'impurity': -np.inf}

        thresholds_l = unique[:(len(unique) - 1)]
        thresholds_h = unique[1:]
        thresholds = (thresholds_l + thresholds_h) / 2
        threshold_best = None
        crit_best = None
        split_results = {}

        for threshold in thresholds:
            right_leaf_X = feature_vector[feature_vector > threshold]
            right_leaf_y = target_vector[feature_vector > threshold]
            left_leaf_X = feature_vector[feature_vector < threshold]
            left_leaf_y = target_vector[feature_vector < threshold]

            # min_samples_leaf
            if (len(right_leaf_y) <= min_samples_leaf) or (len(left_leaf_y) <= min_samples_leaf):
                continue

            unique_y_r = np.unique(right_leaf_y, return_counts=True)
            unique_y_l = np.unique(left_leaf_y, return_counts=True)
            p_r = unique_y_r[1] / len(right_leaf_y)
            p_l = unique_y_l[1] / len(left_leaf_y)
            assert (p_r.sum() == 1) and (p_l.sum() == 1)

            Hr = (p_r * (1 - p_r)).sum()
            Hl = (p_l * (1 - p_l)).sum()

            Q = (- len(right_leaf_y) / len(target_vector) * Hr -
                 len(left_leaf_y) / len(target_vector) * Hl)
            split_results[threshold] = Q

            if (crit_best is None) or (Q > crit_best):
                crit_best = Q
                threshold_best = threshold
                best_split_X = (right_leaf_X, left_leaf_X)
                best_split_y = (right_leaf_y, left_leaf_y)

            elif Q == crit_best:
                old = ((len(best_split_y[0]) / len(target_vector)) ** 2 +
                       (len(best_split_y[1]) / len(target_vector)) ** 2)
                new = ((len(right_leaf_y[0]) / len(target_vector)) ** 2 +
                       (len(left_leaf_y[1]) / len(target_vector)) ** 2)
                if new < old:
                    best_split_X = (right_leaf_X, left_leaf_X)
                    best_split_y = (right_leaf_y, left_leaf_y)

        return {'thresholds': thresholds,
                'ginis': split_results,
                'threshold': threshold_best,
                'impurity': crit_best}

    def _fit_node(self, sub_X, sub_y, node, parent_depth=0):
        node['depth'] = parent_depth + 1
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = sorted(ratio.items(), key=lambda x: x[1], reverse=True)
                categories_map = dict(sorted_categories)
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            # min_samples_split:
            if len(feature_vector) < self._min_samples_split:
                continue

            _, _, threshold, gini = self._find_best_split(feature_vector, sub_y, self._min_samples_leaf).values()
            if threshold is None:
                continue
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold
                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if (feature_best is None) or (node['depth'] > self._max_depth):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], parent_depth=node['depth'])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], parent_depth=node['depth'])


    def _predict_node(self, x, node):
        """
        Recursive function, enters child nodes if node is nonterminal, returns node majority class if node is terminal.
        ___________________________________________________________________________
        :params
            x: numpy array
                Array of feature attributes for each observation
            y: numpy array
                Array of labels for each observation
        """
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        if node['type'] == 'nonterminal':
            nodes_dict = {
                True: self._predict_node(x, node['left_child']),
                False: self._predict_node(x, node['right_child'])
            }
            feature = node['feature_split']
            if self._feature_types[feature] == 'real':
                result = nodes_dict[x[feature] < node['threshold']]
            elif self._feature_types[feature] == 'categorical':
                result = nodes_dict[x[feature] in node['categories_split']]
            else:
                raise ValueError
            return result
        else:
            return node['class']


    def fit(self, X, y):
        """
        Fit algorithm to X and y
        ___________________________________________________________________________
        :params
            X: numpy array
                Array of feature attributes for each observation
            y: numpy array
                Array of labels for each observation
        """
        self._fit_node(X, y, self._tree)


    def predict(self, X):
        """
        Predict label for observation in X
        ___________________________________________________________________________
        :params
            X: numpy array
                Array of feature attributes for each observation

        :returns
            np.array((n, k)), where n - number of observations, k - number of classes.
            Probabilities in range [0, 1].
        """
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def predict_proba(self, X):
        """
        Predict class of each class for observations in X
        ___________________________________________________________________________
        :params
            X: numpy array
                Array of feature attributes for each observation

        :returns
            np.array((n, k)), where n - number of observations, k - number of classes.
            Probabilities in range [0, 1].
        """
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)


class DecisionTreeRegressor(DecisionTreeClassifier):
    """
    Decision tree classifier
    ____________________________________________________________________________
    :params
        feature_types: list
            List o feature types ('num' or 'categorical'). Determines encoding
        max_depth: int/float, default=+inf
            Maximum depth of the tree.
        min_samples_split: int, default=2
            Minimum samples requirement for a parent node to be split
        min_samples_leaf: int, default=1
            Minimum samples requirement in a child node to be considered
    """
    def __init__(self, feature_types, max_depth=np.inf, min_samples_split=2, min_samples_leaf=1):
        super().__init__(feature_types, max_depth, min_samples_split, min_samples_leaf)

    def _fit_node(self, sub_X, sub_y, node, parent_depth=0):
        node['depth'] = parent_depth + 1
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = sorted(ratio.items(), key=lambda x: x[1], reverse=True)
                categories_map = dict(sorted_categories)
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

        if (feature_best is None) or (node['depth'] > self._max_depth):
            node["type"] = "terminal"
            node["class"] = np.average(sub_y)
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], parent_depth=node['depth'])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], parent_depth=node['depth'])
