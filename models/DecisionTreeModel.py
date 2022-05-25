from random import seed, randint
from statistics import mean, stdev

import pydotplus
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from io import StringIO
from IPython.display import Image
from imblearn.metrics import sensitivity_score, specificity_score

from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import decomposition, datasets

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_fscore_support, \
    roc_curve, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from models.Model import Model
from options.decision_tree_options import DecisiontreeOptions
from utils.utils import countLabels


def draw_results(feature_cols, clf):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('sams-medication.png')
    Image(graph.create_png())


class DecisionTreeModel(Model):
    def __init__(self, data, options: DecisiontreeOptions):
        super().__init__(data)
        self.options = options

        #    X, y = make_imbalance(
        #        data[options.feature_col],
        #        data[options.target_col],
        #        sampling_strategy={0: 25, 1: 50, 2: 50},
        #        random_state=options.random_state,
        #    )
        self.X = data[options.feature_col]  # Features
        self.y = data[options.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                train_size=options.train_size,
                                                                                random_state=options.random_state,
                                                                                stratify=self.y)
        # Create Decision Tree classifer object
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        self.clf = DecisionTreeClassifier(criterion=self.options.criterion, max_depth=self.options.max_depth,
                                          max_features=self.options.max_features, splitter=self.options.splitter,
                                          random_state=options.random_state, class_weight=self.options.class_weight)

    def cross_validate(self, folds):
        cvs = cross_val_score(self.clf, self.X, self.y, cv=folds)
        print(cvs)

    # ccp_alpha unbalanced: 0.0175
    # ccp_alpha balanced: 0
    def pruning(self):
        clf = DecisionTreeClassifier(random_state=self.options.random_state, class_weight=self.options.class_weight)
        path = clf.cost_complexity_pruning_path(self.X_train, self.y_train)

        # ccp_alphas: alfas waar tree verandert (Tree score = SSR)

        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        fig, ax = plt.subplots()
        ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
        ax.set_xlabel("effective alpha")
        ax.set_ylabel("total impurity of leaves")
        ax.set_title("Total Impurity vs effective alpha for training set")
        plt.show()

        # train trees with these alphas, max test= 0.0175
        clfs = []
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha, class_weight=self.options.class_weight)
            clf.fit(self.X_train, self.y_train)
            clfs.append(clf)

        # look for best alpha
        train_scores = [clf.score(self.X_train, self.y_train) for clf in clfs]
        test_scores = [clf.score(self.X_test, self.y_test) for clf in clfs]

        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs alpha for training and testing sets")
        ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
        ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
        ax.legend()
        plt.show()

    # not finished
    def gridsearch(self):
        scaler = StandardScaler()
        clf = DecisionTreeClassifier(class_weight=self.options.class_weight)

        pipe = Pipeline(steps=[('std_slc', scaler),
                               ('dec_tree', clf)])

        criterion = ['entropy', 'gini']
        max_depth = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]

        parameters = dict(dec_tree__criterion=criterion,
                          dec_tree__max_depth=max_depth)

        clf_GS = GridSearchCV(pipe, parameters)
        clf_GS.fit(self.X, self.y)

        print('Best Criterion:', clf_GS.best_estimator_.get_params()['dec_tree__criterion'])
        print('Best max_depth:', clf_GS.best_estimator_.get_params()['dec_tree__max_depth'])
        print(clf_GS.best_estimator_.get_params()['dec_tree'])

    # abstract override
    def scale_model(self, scaler):
        scaler.fit(self.X_train)
        Standardized_X_train = scaler.transform(self.X_train)
        Standardized_X_test = scaler.transform(self.X_test)

        # necessary?
        self.X_train = Standardized_X_train
        self.X_test = Standardized_X_test

    # abstract override
    def train_model(self):
        # Train Decision Tree Classifer
        self.clf = self.clf.fit(self.X_train, self.y_train)

    def confusion_matrix(self, y_pred):
        cfm = confusion_matrix(y_true=self.y_test, y_pred=y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cfm)
        disp.plot()
        plt.show()

    # abstract override
    def test_model(self, isMultilabel: bool):
        # Predict the response for test dataset
        train_pred = self.clf.predict(self.X_train)
        y_pred = self.clf.predict(self.X_test)

        # Model Accuracy, how often is the classifier correct?
        print("Accuracy on train data:", metrics.accuracy_score(self.y_train, train_pred))
        print("Accuracy on test data:", metrics.accuracy_score(self.y_test, y_pred))

        if not isMultilabel:
            # gives sometimes warning : UserWarning: y_pred contains classes not in y_true
            #   warnings.warn("y_pred contains classes not in y_true")
            # you cant use this fnct on multilabel data
            print("Balanced accuracy on train data:", metrics.balanced_accuracy_score(self.y_train, train_pred))
            print("Balanced accuracy on test data:", metrics.balanced_accuracy_score(self.y_test, y_pred))

        # confusion matrix
        self.calc_mean_std(10)
        # self.confusion_matrix(y_pred)
        # self.f1_score(y_pred)
        # self.sensitivity_score(y_pred)
        # self.specificity_score(y_pred)
        # self.precision_recall_fscore_support(y_pred)
        # self.plot_roc_curve(y_pred)

    def precision_recall_fscore_support(self, y_pred):
        res = []
        for l in [0, 1, 2]:
            prec, recall, _, _ = precision_recall_fscore_support(np.array(self.y_test) == l,
                                                                 np.array(y_pred) == l,
                                                                 pos_label=True, average=None)
            res.append([l, recall[1], recall[0]])
        res = pd.DataFrame(res, columns=['class', 'sensitivity', 'specificity'])
        print(res)

    # f1 score
    def f1_score(self, y_pred):
        f1score = f1_score(self.y_test, y_pred, average='weighted')  # what does average mean??
        print('weighted f1_score: ' + str(f1score))

    def sensitivity_score(self, y_pred):
        sensScore = sensitivity_score(self.y_test, y_pred, average='weighted')
        print('weighted sensitivity: ' + str(sensScore))

    def specificity_score(self, y_pred):
        sensScore = specificity_score(self.y_test, y_pred, average='weighted')
        print('weighted specificity: ' + str(sensScore))

    # count amount of labels in prediction and actual amount of labels
    # used for evaluating test/train size
    # irrelevant now, just use stratify in data splitter
    def count_test_split(self, toPrint: bool):
        y_pred = self.clf.predict(self.X_test)
        resultLabels = countLabels(y_pred)
        testLabels = countLabels(self.y_test)
        if toPrint:
            print("Number of labels found in prediction model: " + str(len(resultLabels.keys())))
            print("Number of labels in test data: " + str(len(testLabels.keys())))
        return [len(testLabels.keys()), len(resultLabels.keys())]

    def calc_mean_std(self, it):
        totalres = [0, 0, 0]
        sens = {0: [], 1: [], 2: []}
        spec = {0: [], 1: [], 2: []}
        y_pred = list()
        seed(2)
        for i in range(it):

            self.options.set_random_state(randint(0, 999))
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                                random_state=self.options.random_state,
                                                                train_size=self.options.train_size,
                                                                stratify=self.y)

            clf = DecisionTreeClassifier(criterion=self.options.criterion, max_depth=self.options.max_depth,
                                         max_features=self.options.max_features, splitter=self.options.splitter,
                                         random_state=self.options.random_state, class_weight=self.options.class_weight)
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
        print(stdev(sens[0]))

    # plot graph depth - train/test accuracy
    def plot_depth(self, max_depth, isMultilabel):

        train_accuracy = list()
        test_accuracy = list()

        train_balanced_accuracy = list()
        test_balanced_accuracy = list()

        for depth in range(1, max_depth):
            clf = DecisionTreeClassifier(criterion=self.options.criterion, max_depth=depth,
                                         max_features=self.options.max_features, splitter=self.options.splitter,
                                         random_state=self.options.random_state)

            clf.fit(self.X_train, self.y_train)
            train_pred = clf.predict(self.X_train)
            test_pred = clf.predict(self.X_test)

            train_accuracy.append(metrics.accuracy_score(self.y_train, train_pred))
            test_accuracy.append(metrics.accuracy_score(self.y_test, test_pred))

            if not isMultilabel:
                train_balanced_accuracy.append(metrics.balanced_accuracy_score(self.y_train, train_pred))
                test_balanced_accuracy.append(metrics.balanced_accuracy_score(self.y_test, test_pred))

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.plot(range(1, max_depth), train_accuracy, color='tab:blue')
        ax1.plot(range(1, max_depth), test_accuracy, color='tab:orange')

        ax1.set_title('Train vs test accuracy with variable depth')

        plt.show()

        if not isMultilabel:
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(1, 1, 1)
            ax2.plot(range(1, max_depth), train_balanced_accuracy, color='tab:blue')
            ax2.plot(range(1, max_depth), test_balanced_accuracy, color='tab:orange')

            ax2.set_title('Train vs test balanced accuracy with variable depth')

            plt.show()

    def plot_roc_curve(self, y_pred):
        fpr, tpr, _ = roc_curve(self.y_test, y_pred, pos_label=self.clf.classes_[2])
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.show()
