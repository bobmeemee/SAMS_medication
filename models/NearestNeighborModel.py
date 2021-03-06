from random import seed, randint
from statistics import mean, stdev

import numpy as np
import pandas as pd
from imblearn.metrics import sensitivity_score, specificity_score
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support, \
    accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from models.Model import Model
from options.nearest_neighbor_options import NearestNeighborOptions


class NearestNeighborModel(Model):
    def __init__(self, data, options: NearestNeighborOptions):
        super().__init__(data)
        self.options = options

        self.X = data[options.feature_col]  # Features
        self.y = data[options.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                train_size=options.train_size,
                                                                                random_state=options.random_state,
                                                                                stratify=self.y)

        self.clf = KNeighborsClassifier(n_neighbors=self.options.n_neighbors, metric=options.metric, p=options.p,
                                        n_jobs=options.n_jobs)

    def scale_model(self, scaler):
        scaler.fit(self.X_train)
        Standardized_X_train = scaler.transform(self.X_train)
        Standardized_X_test = scaler.transform(self.X_test)

        self.X_train = Standardized_X_train
        self.X_test = Standardized_X_test

    def train_model(self):
        self.clf = self.clf.fit(self.X_train, self.y_train)

    def test_model(self, isMultilabel: bool):
        print("Accuracy: " + str(self.clf.score(self.X_test, self.y_test)))
        y_pred = self.clf.predict(self.X_test)
        self.calc_mean_std(10)

        # self.confusion_matrix(y_pred)
        # print('f1_score: ' + str(self.f1_score(y_pred)))
        # print('weighted sensitivity: ' + str(self.sensitivity_score(y_pred)))
        # print('weighted specifity: ' + str(self.specificity_score(y_pred)))
        # self.precision_recall_fscore_support(y_pred)
        # self.test_k()

    # draw cfm of predicted data y_pred
    def confusion_matrix(self, y_pred):
        cfm = confusion_matrix(y_true=self.y_test, y_pred=y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cfm)
        disp.plot()
        plt.show()

    # calculate weighted f1score
    def f1_score(self, y_pred):
        return f1_score(self.y_test, y_pred, average='weighted')  # what does average mean??

    # calculate weighted sensitivity
    def sensitivity_score(self, y_pred):
        return sensitivity_score(self.y_test, y_pred, average='weighted')

    # calculate weighted specificity
    def specificity_score(self, y_pred):
        return specificity_score(self.y_test, y_pred, average='weighted')

    # only works for 3x3
    # calc recall and spec for each label
    def precision_recall_fscore_support(self, y_pred):
        res = []
        for l in [0, 1, 2]:
            prec, recall, _, _ = precision_recall_fscore_support(np.array(self.y_test) == l,
                                                                 np.array(y_pred) == l,
                                                                 pos_label=True, average=None)
            res.append([l, recall[1], recall[0]])
        res = pd.DataFrame(res, columns=['class', 'sensitivity', 'specificity'])
        print(res)

    # scale model has to be done before this function!
    # recall & specificty for 25 iterations of k
    def test_k(self):
        for k in range(1, 25):
            knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
            knn.fit(self.X_train, self.y_train)
            print(str(k) + ": " + str(knn.score(self.X_test, self.y_test)))
            self.precision_recall_fscore_support(knn.predict(self.X_test))


    # calc mean and std dev for it iterations
    def calc_mean_std(self, it):
        totalres = [0, 0, 0]
        sens = {0: [], 1: [], 2: []}
        spec = {0: [], 1: [], 2: []}
        accuracy = list()
        # seed(2)
        for i in range(it):

            self.options.set_random_state(randint(0, 999))
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                                random_state=self.options.random_state,
                                                                train_size=self.options.train_size,
                                                                stratify=self.y)
            scaler = StandardScaler()
            scaler.fit(X_train)
            Standardized_X_train = scaler.transform(self.X_train)
            Standardized_X_test = scaler.transform(self.X_test)

            X_train = Standardized_X_train
            X_test = Standardized_X_test

            clf = KNeighborsClassifier(n_neighbors=self.options.n_neighbors, metric=self.options.metric,
                                       p=self.options.p,
                                       n_jobs=self.options.n_jobs, weights='uniform')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy.append(accuracy_score(y_test, y_pred))

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
              "stdev: " + str(stdev(accuracy)) + "\n")

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
