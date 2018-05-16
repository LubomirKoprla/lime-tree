import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
import pandas as pd
from sklearn.utils import check_random_state

from sklearn.metrics.pairwise import manhattan_distances

class limeTree_Text:
    """Explains predictions on text data."""
    
    def __init__(self,
                 predict_fn,
                 vectorizer,
                 features_names=None):
        """Init function.

        Args:
            predict_fn: prediction function. This should be a
                function that takes a numpy array and outputs prediction
                probabilities. For ScikitClassifiers, this is
                `classifier.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            features_names: list of names (strings) corresponding to the features 
                in the model
            vectorizer: scikit learn class TfidfVectorizer for text procesing
        """
        if features_names is None:
            self.features_names = vectorizer.vocabulary_.items()
        else:
            self.features_names = list(features_names)
            
        self.predict_fn = predict_fn
        self.vectorizer = vectorizer
        
    def explain(self,
                text,
                tree_classifier = None,
                samples = 1000,
                get_distances = manhattan_distances,
                random_state = None):
        
        if tree_classifier is None:
            tree_classifier = DecisionTreeClassifier()
        random_state = check_random_state(random_state)
        text_row = text.split()
        text_size = len(text_row)
        data = np.ones((samples, text_size))
        num_samples = random_state.randint(1, text_size + 1, samples - 1)
        features_range = range(text_size)
        for i, size in enumerate(num_samples, start=1):
            inactive = random_state.choice(features_range, size-1,
                                                replace=False)
            data[i, inactive] = 0
            
        data_text = []
        for row in data:
            word = None
            for i,flag in enumerate(row):
                if flag == 1:
                    if word == None:
                        word = text_row[i]
                    else:
                        word = word + " " + text_row[i]
            data_text.append(word)
        labels = self.predict_fn(self.vectorizer.transform(data_text))
        self.labels = labels
        weights = 1 / (get_distances([data[0]], data) + 1)
        tree_classifier.fit(self.vectorizer.transform(data_text), labels, sample_weight=weights[0, :])
        self._tree = tree_classifier
        self.rules = self.__tree_to_rules(self.vectorizer.transform([text]))
        return self.rules
    
    def __tree_to_rules(self,row,precision=2):
        """Transform decision tree to rules

        Args:
            row: 1d numpy array, corresponding to a row.
            precision: precision of output values. Default is 2 digits after the decimal point.
        Returns:
            rules: 1d array with rules in string.
        """

        tree_ = self._tree.tree_
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
                if row[:,feature_num] <= threshold:
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
                    importance = round(self._tree.feature_importances_[i],precision)
                    break
            if key not in greater:
                rules.append("{} menej krát, importance {}".format(key, importance))
            else:
                rules.append("{} stredne veľa krát, importance {}".format(key, importance))
        for key, value in greater.items():
            if key not in smaller:
                for i, x in enumerate(self.features_names):
                    if (x == key):
                        importance = round(self._tree.feature_importances_[i], precision)
                        break
                rules.append("{} viac krát, importance {}".format(key, importance))
        return rules