from random import randint, seed
from statistics import mean, stdev

import pydotplus
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report, \
    precision_recall_fscore_support, roc_curve, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.metrics import sensitivity_score, specificity_score

from models.Model import Model
from options.random_forest_options import RandomForestOptions


class RandomForestModel(Model):
    def __init__(self, data, options: RandomForestOptions):
        super().__init__(data)
        self.options = options

        self.X = data[options.feature_col]  # Features
        self.y = data[options.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=options.train_size,
                                                                                random_state=options.random_state,
                                                                                stratify=self.y)

        if options.balancedRFC:
            self.clf = BalancedRandomForestClassifier(criterion=self.options.criterion,
                                                      n_estimators=self.options.n_estimators,
                                                      max_depth=self.options.max_depth,
                                                      max_features=self.options.max_features,
                                                      random_state=self.options.random_state,
                                                      class_weight=self.options.class_weight)
        else:
            self.clf = RandomForestClassifier(criterion=self.options.criterion, n_estimators=self.options.n_estimators,
                                              max_depth=self.options.max_depth, max_features=self.options.max_features,
                                              random_state=self.options.random_state)

    def scale_model(self, scaler):
        pass

    def setBalancedRandomForestClassifier(self):
        self.clf = BalancedRandomForestClassifier(criterion=self.options.criterion,
                                                  n_estimators=self.options.n_estimators,
                                                  max_depth=self.options.max_depth,
                                                  max_features=self.options.max_features,
                                                  random_state=self.options.random_state,
                                                  class_weight=self.options.class_weight)

    def setRandomForestClassifier(self):
        self.clf = RandomForestClassifier(criterion=self.options.criterion, n_estimators=self.options.n_estimators,
                                          max_depth=self.options.max_depth, max_features=self.options.max_features,
                                          random_state=self.options.random_state)

    def train_model(self):
        self.clf = self.clf.fit(self.X_train, self.y_train)

    def test_model(self, isMultilabel: bool):
        y_pred = self.clf.predict(self.X_test)

        score = self.clf.score(self.X_test, self.y_test)
        print("Accuray: " + str(score))
        self.calc_mean_std(10)
        # self.confusion_matrix(y_pred)
        # print('f1_score: ' + str(self.f1_score(y_pred)))
        # print('weighted selectivity: ' + str(self.sensitivity_score(y_pred)))
        # print('weighted specifity: ' + str(self.specificity_score(y_pred)))
        # self.precision_recall_fscore_support(y_pred)
        # self.plot_roc_curve(y_pred)
        # print(classification_report(self.y_test, y_pred))

    def confusion_matrix(self, y_pred):
        cfm = confusion_matrix(y_true=self.y_test, y_pred=y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cfm)
        disp.plot()
        plt.show()

    def f1_score(self, y_pred):
        return f1_score(self.y_test, y_pred, average='weighted')  # what does average mean??

    def sensitivity_score(self, y_pred):
        return sensitivity_score(self.y_test, y_pred, average='weighted')

    def specificity_score(self, y_pred):
        return specificity_score(self.y_test, y_pred, average='weighted')

    # only works for 3x3
    def precision_recall_fscore_support(self, y_pred):
        res = []
        for l in [0, 1, 2]:
            prec, recall, _, _ = precision_recall_fscore_support(np.array(self.y_test) == l,
                                                                 np.array(y_pred) == l,
                                                                 pos_label=True, average=None)
            res.append([l, recall[1], recall[0]])
        res = pd.DataFrame(res, columns=['class', 'sensitivity', 'specificity'])
        print(res)

    def plot_roc_curve(self, y_pred):
        fpr, tpr, _ = roc_curve(self.y_test, y_pred, pos_label=self.clf.classes_[2])
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.show()

    def calc_mean_std(self, it):
        totalres = [0, 0, 0]
        sens = {0: [], 1: [], 2: []}
        spec = {0: [], 1: [], 2: []}
        accuracy = list()
        for i in range(it):
            self.options.set_random_state(randint(0, 999))

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                                random_state=self.options.random_state,
                                                                train_size=self.options.train_size,
                                                                stratify=self.y)

            clf = BalancedRandomForestClassifier(criterion=self.options.criterion, max_depth=self.options.max_depth,
                                                 max_features=self.options.max_features,
                                                 random_state=self.options.random_state,
                                                 class_weight=self.options.class_weight)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy.append(metrics.accuracy_score(y_test, y_pred))
            res = []
            for l in [0, 1, 2]:
                prec, recall, fscore, support = precision_recall_fscore_support(np.array(y_test) == l,
                                                                                np.array(y_pred) == l,
                                                                                pos_label=True, average=None)
                sens[l].append(recall[1])
                spec[l].append(recall[0])
                res.append([l, recall[1], recall[0]])
            totalres = np.add(totalres, res)
        totalres = np.divide(totalres, it)
        totalres = pd.DataFrame(totalres, columns=['class', 'sensitivity', 'specificity'])
        print(totalres)

        print("Number of iterations: " + str(it))

        print("Mean accuracy: " + str(mean(accuracy)) +
              " stdev: " + str(stdev(accuracy)) + "\n")

        meanSens = list()
        for key, value in sens.items():
            meanSens.append([key, mean(value), stdev(value)])
        meanSens = pd.DataFrame(meanSens, columns=['class', 'mean sensitivity', 'standard deviation'])
        print(meanSens)

        meanSpec = list()
        for key, value in spec.items():
            meanSpec.append([key, mean(value), stdev(value)])
        meanSpec = pd.DataFrame(meanSpec, columns=['class', 'mean specificity', 'standard deviation'])
        print(meanSpec)
