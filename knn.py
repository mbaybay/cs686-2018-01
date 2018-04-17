from classifier import classifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class KNN(classifier):
    def __init__(self, k=3):
        self.x = None
        self.y = None
        self.k = k

    def fit(self, x, y):
        """

        :param X: pandas data frame, with rows as observations and columns as observation features
        :param Y: pandas series, encoded as categorical classes
        :return:
        """
        # compute distances
        self.x = x
        self.y = y

    def predict(self, x):
        """

        :param x:
        :return:
        """
        pred = x.apply(self.__get_majority_class, axis=1)
        return pred.tolist()

    def __get_majority_class(self, obs):
        """

        :param obs: one observation of test x
        :return:
        """
        # distances = pd.Series(index=self.x.index)

        def __calc_euclidean_distance(row):
            # https://en.wikipedia.org/wiki/Euclidean_distance
            # d(p,q) = sqrt(sum((p_i-q_i)**2))
            euc_dist = np.sqrt(np.square(row.subtract(obs)).sum())  # row - sub
            return euc_dist

        # calculate euclidean distance to all points
        distances = self.x.apply(func=__calc_euclidean_distance, axis=1)

        # sort by distance
        distances.sort_values(ascending=False, inplace=True)

        # take top-k observations by index
        top_k = distances[0:self.k].index

        # get majority class from y
        majority_class = self.y[top_k].value_counts()
        majority_class = majority_class.index[0]
        return majority_class


if __name__ == '__main__':
    import util

    data = util.read_arff_as_df("../../data/PhishingData.arff.txt")
    x = data.loc[:, data.columns != 'Result']
    y = data['Result']

    train_x, test_x, train_y, test_y = train_test_split(x, y)

    clf = KNN(k=3)
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    util.print_confusion_matrix(test_y, pred, labels=[-1, 0, 1])
