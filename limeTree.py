import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
import pandas as pd

class limeTree:
    """Explains predictions on tabular numerical data."""

    def __init__(self,
                 training_data,
                 predict_fn,
                 discrete_features=None,
                 features_names=None):
        """Init function.

        Args:
            training_data: numpy 2d array.
            predict_fn: prediction function. This should be a
                function that takes a numpy array and outputs prediction
                probabilities. For ScikitClassifiers, this is
                `classifier.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            discrete: list of indices (ints) corresponding to the
                categorical(discrete) columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            features_names: list of names (strings) corresponding to the columns
                in the training data.
        """
        if discrete_features is None:
            self.discrete_features = []
        else:
            self.discrete_features = list(discrete_features)
        if features_names is None:
            self.features_names = list(range(training_data.shape[1]))
        else:
            self.features_names = list(features_names)

        # proces numerical atributes
        scaler = StandardScaler(with_mean=False)
        scaler.fit(training_data)
        self.data_mean = scaler.mean_
        self.data_std = scaler.scale_
        self.training_data = training_data
        self.predict_fn = predict_fn

    def explain(self,
                data_row,
                tree_classifier = None,
                generate_data = True,
                samples = 100,
                generate_data_ratio = 0.7,
                precision = 2,
                get_dist = manhattan_distances):

        """Generates explanations for a prediction.

        First, it generates a neighborhood around a prediction, model makes predictions on this sample and on this data it trains tree_classifier and
        return rules for given row on given model.

        Args:
            data_row: 1d numpy array, corresponding to a row.
            tree_classifier: sklearn DecisionTreeClasifier, default is with default model parameters.
                You can add DecisionTreeClasifier with cwn parameters.
            generate_data: boolean value, if you want generate data(True), or you want choose from train dataset(false).
            samples: size of the neighborhood to learn the Decision Tree classifier.
            generate_data_ratio: ratio of data generate around prediction and from training data.
            precision: precision of output values. Default is 2 digits after the decimal point.
            get_distances: function for calcute distances between row and rows in data. E.g. sklearn.metrics.pairwise.manhattan_distances.

        Returns
        """

        if generate_data:
            self.data = self.__data_generator(data_row, samples, generate_data_ratio)
        else:
            self.data = self.__data_choice(samples)

        if tree_classifier is None:
            tree_classifier = DecisionTreeClassifier()


            # make prediction on generated data
        self.data_pred = self.predict_fn(self.data)

        # weighting generated data
        self.weights = self.__get_weights(data_row, self.data, get_dist)
        tree_classifier.fit(self.data, self.data_pred, sample_weight=self.weights[0, :])
        self.tree_classifier = tree_classifier
        rules = self.__tree_to_rules(data_row,precision)
        return rules

    def get_tree(self):
        """After calling method explain, you can get tree classifier, which was trained.

        Returns
            Trained sklearn.tree.DecisionTreeClassifier
        """

        return self.tree_classifier

    def __data_generator(self,
                         data_row,
                         samples,
                         generate_data_ratio):
        """Generates a neighborhood around a prediction.

        For numerical features, it generate (generate_data_ratio*samples) data from Normal(mean,std), where mean is value from data_row and
        std is std in the training data. After that, it generate ((1-generate_data_ratio)*samples) data from Normal(mean,std) , where mean and std, according to
        the means and stds in the training data.
        For categorical features, it random choose data  from training data.

        Args:
            data_row: 1d numpy array, corresponding to a row.
            samples: size of the neighborhood to learn the linear model.
            generate_data_ratio: how many samples are generate from the neighborhood.
        Returns:
            gen_data: samples * features number matrix, neighborhood around a prediction.
        """
        row_distr_samples = int(samples * generate_data_ratio)
        data_distr_samples = int(samples * (1 - generate_data_ratio))
        samples = row_distr_samples + data_distr_samples
        gen_data = []
        for i in range(len(self.features_names)):
            if i in self.discrete_features:
                values = np.random.choice(self.training_data[:, i], size=samples)
                gen_data.append(values)

            else:
                values_around_row = np.random.normal(loc=data_row[i], scale=self.data_std[i],
                                                     size=row_distr_samples)
                values_from_data = np.random.normal(loc=self.data_mean[i], scale=self.data_std[i],
                                                    size=data_distr_samples)

                gen_data.append(np.concatenate((values_around_row, values_from_data)))

        gen_data = np.transpose(gen_data)
        test = pd.DataFrame(gen_data)
        print(test.shape)
        return gen_data

    def __data_choice(self,
                      samples):
        """Randomly choose neighborhood around a prediction from training data.

        Args:
            samples: size of the neighborhood to learn the linear model.

        Returns:
            gen_data: samples * features number matrix, neighborhood around a prediction.
        """

        new_data = self.training_data[np.random.choice(list(range(self.training_data.shape[0])), size=samples), :]
        return new_data

    def __get_weights(self,
                      data_row,
                      data,
                      get_distances):
        """It sets weights for samples in data. Weights is from manhattan distance between explanation and row in data and
         closer rows have bigger weights.

        Args:
            data_row: 1d numpy array, corresponding to a row.
            data: 2d array, samples * features number matrix.

        Returns:
            weights: 1d numpy array with weights of rows in data
        """

        weights = 1 / (get_distances([data_row], data) + 1)
        return weights

    def __tree_to_rules(self,row,precision):
        """Transform decision tree to rules

        Args:
            row: 1d numpy array, corresponding to a row.
            precision: precision of output values. Default is 2 digits after the decimal point.
        Returns:
            weights: 1d array with rules in string.
        """

        tree_ = self.tree_classifier.tree_
        feature_name = [
            self.features_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        smaller = {}
        greater = {}

        def recurse(node, depth):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                for i, x in enumerate(self.features_names):
                    if (x == name):
                        feature_num = i
                        break
                if row[feature_num] <= threshold:
                    smaller[name] = threshold
                    recurse(tree_.children_left[node], depth + 1)
                else:
                    greater[name] = threshold
                    recurse(tree_.children_right[node], depth + 1)

        recurse(0, 1)
        rules = []
        for key, value in smaller.items():
            for i, x in enumerate(self.features_names):
                if (x == key):
                    importance = round(self.tree_classifier.feature_importances_[i],precision)
                    break
            if key not in greater:
                rules.append("{} <= {}, importance {}".format(key, round(value,precision), importance))
            else:
                rules.append("{} < {} <= {}, importance {}".format(round(greater[key],precision), key, round(value,precision), importance))
        for key, value in greater.items():
            if key not in smaller:
                for i, x in enumerate(self.features_names):
                    if (x == key):
                        importance = round(self.tree_classifier.feature_importances_[i], precision)
                        break
                rules.append("{} > {}, importance {}".format(key, round(value,precision), importance))
        return rules