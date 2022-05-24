import pydotplus
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

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

        X = data[options.feature_col]  # Features
        y = data[options.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=options.train_size,
                                                                                random_state=options.random_state,
                                                                                stratify=y)

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
        scaler.fit(self.X_train)
        Standardized_X_train = scaler.transform(self.X_train)
        Standardized_X_test = scaler.transform(self.X_test)

        self.X_train = Standardized_X_train
        self.X_test = Standardized_X_test

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
        self.confusion_matrix(y_pred)
        print('f1_score: ' + str(self.f1_score(y_pred)))
        print('weighted selectivity: ' + str(self.sensitivity_score(y_pred)))
        print('weighted specifity: ' + str(self.specificity_score(y_pred)))
        self.precision_recall_fscore_support(y_pred)
        self.plot_roc_curve(y_pred)
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
