import numpy as np
import pandas as pd
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
        X_one_hot = pd.DataFrame(encoder.fit_transform(X_data[options.feature_col]).toarray())

        X = X_one_hot  # Features as one hot
        y = data[options.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=options.train_size,
                                                                                random_state=options.random_state,
                                                                                stratify=y)

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
