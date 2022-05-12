import pydotplus
import matplotlib.pyplot as plt
from io import StringIO
from IPython.display import Image

from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from imblearn.datasets import make_imbalance

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
        X = data[options.feature_col]  # Features
        y = data[options.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=options.train_size,
                                                                                random_state=options.random_state,
                                                                                stratify=y)
        # Create Decision Tree classifer object
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        self.clf = DecisionTreeClassifier(criterion=self.options.criterion, max_depth=self.options.max_depth,
                                          max_features=self.options.max_features, splitter=self.options.splitter,
                                          random_state=options.random_state)

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

    def confusion_matrix(self):
        y_pred = self.clf.predict(self.X_test)

        cfm = confusion_matrix(y_true=self.y_test, y_pred=y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cfm)
        disp.plot()
        plt.show()

    def f1_score(self):
        y_pred = self.clf.predict(self.X_test)
        f1_score(y_true=self.y_test, y_pred=y_pred)

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
        self.confusion_matrix()

        # f1 score

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
