from random import randint
from statistics import mean

import numpy as np
import pandas as pd
from imblearn import metrics
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from models.Model import Model
from options.logistic_regression_options import LogisticRegressionOptions


class LogisticRegressionModel(Model):
    def __init__(self, data, options: LogisticRegressionOptions):
        super().__init__(data)
        self.options = options
        # one hot encoding necessary
        # creating instance of one-hot-encoder
        encoder = OneHotEncoder(handle_unknown='ignore')

        # perform one-hot encoding on feature columns column
        X_data = data[options.feature_col]
        self.X_one_hot = pd.DataFrame(encoder.fit_transform(X_data[options.feature_col]).toarray())

        self.X = self.X_one_hot  # Features as one hot
        self.y = data[options.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                train_size=options.train_size,
                                                                                random_state=options.random_state,
                                                                                stratify=self.y)

        self.clf = LogisticRegression(random_state=options.random_state, class_weight=options.class_weight)

    # not necessary with logistic regression
    def scale_model(self, scaler):
        pass

    def train_model(self):
        self.clf.fit(self.X_train, self.y_train)
        pass

    def test_model(self, isMultilabel: bool):
        print("Score: " + str(self.clf.score(self.X_test, self.y_test)))

        y_pred = self.clf.predict(self.X_test)
        self.confusion_matrix(y_pred)
        self.precision_recall_fscore_support(y_pred)
        self.calc_mean_std(10)
        # self.plot_train_size(20, 80, 5)

    def confusion_matrix(self, y_pred):
        cfm = confusion_matrix(y_true=self.y_test, y_pred=y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cfm)

        disp.plot()
        plt.show()

    def precision_recall_fscore_support(self, y_pred):
        res = []
        for i in [0, 1, 2]:
            prec, recall, _, _ = precision_recall_fscore_support(np.array(self.y_test) == i,
                                                                 np.array(y_pred) == i,
                                                                 pos_label=True, average=None)
            res.append([i, recall[1], recall[0]])
        res = pd.DataFrame(res, columns=['class', 'recall', 'specificity'])
        print(res)

    def plot_train_size(self, min_train_size, max_train_size, step_size):
        train_accuracy = list()
        test_accuracy = list()

        for ts in range(min_train_size, max_train_size, step_size):
            print(ts)
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                                train_size=ts / 100,
                                                                random_state=self.options.random_state,
                                                                stratify=self.y)

            local_clf = LogisticRegression(random_state=self.options.random_state,
                                           class_weight=self.options.class_weight)
            local_clf.fit(X_train, y_train)

            train_accuracy.append(local_clf.score(X_train, y_train))
            test_accuracy.append(local_clf.score(X_test, y_test))

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        ax1.plot(range(min_train_size, max_train_size, step_size), train_accuracy, color='tab:blue')
        ax1.plot(range(min_train_size, max_train_size, step_size), test_accuracy, color='tab:orange')
        ax1.set_title('Train vs test accuracy with variable train size')
        plt.show()

    def calc_mean_std(self, it):
        totalres = [0, 0, 0]
        sens = {0: [], 1: [], 2: []}
        spec = {0: [], 1: [], 2: []}
        y_pred = list()
        # seed(1) # makes results reproducable
        for i in range(it):
            self.options.set_random_state(randint(0, 999))

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                                random_state=self.options.random_state,
                                                                train_size=self.options.train_size,
                                                                stratify=self.y)

            clf = LogisticRegression(random_state=self.options.random_state,
                                     class_weight=self.options.class_weight)
            clf.fit(X_train, y_train)
            y_pred.append(clf.predict(X_test))

            res = []
            for l in [0, 1, 2]:
                prec, recall, fscore, support = precision_recall_fscore_support(np.array(y_test) == l,
                                                                                np.array(y_pred[i]) == l,
                                                                                pos_label=True, average=None)
                sens[l].append(recall[1])
                spec[l].append(recall[0])
                res.append([l, recall[1], recall[0]])
            totalres = np.add(totalres, res)
        totalres = np.divide(totalres, it)
        totalres = pd.DataFrame(totalres, columns=['class', 'sensitivity', 'specificity'])
        print(totalres)
        print(mean(sens[0]))
        print(mean(spec[0]))
