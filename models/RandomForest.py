import pydotplus
from matplotlib import pyplot as plt

from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from imblearn.ensemble import BalancedRandomForestClassifier

from models.Model import Model
from options.random_forest_options import RandomForestOptions


class RandomForestModel(Model):
    def __init__(self, data, options: RandomForestOptions):
        super().__init__(data)
        self.options = options

        X = data[options.feature_col]  # Features
        y = data[options.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=options.train_size,
                                                                                random_state=options.random_state)
        if options.balancedRFC:
            self.clf = BalancedRandomForestClassifier(criterion=self.options.criterion,
                                                      n_estimators=self.options.n_estimators,
                                                      max_depth=self.options.max_depth,
                                                      max_features=self.options.max_features,
                                                      random_state=self.options.random_state)
        else:
            self.setRandomForestClassifier()

    def scale_model(self, scaler):
        scaler.fit(self.X_train)
        Standardized_X_train = scaler.transform(self.X_train)
        Standardized_X_test = scaler.transform(self.X_test)

        # necessary?
        self.X_train = Standardized_X_train
        self.X_test = Standardized_X_test

    def setBalancedRandomForestClassifier(self):
        self.clf = BalancedRandomForestClassifier(criterion=self.options.criterion,
                                                  n_estimators=self.options.n_estimators,
                                                  max_depth=self.options.max_depth,
                                                  max_features=self.options.max_features,
                                                  random_state=self.options.random_state)

    def setRandomForestClassifier(self):
        self.clf = RandomForestClassifier(criterion=self.options.criterion, n_estimators=self.options.n_estimators,
                                          max_depth=self.options.max_depth, max_features=self.options.max_features,
                                          random_state=self.options.random_state)

    def train_model(self):
        self.clf = self.clf.fit(self.X_train, self.y_train)

    def test_model(self, isMultilabel: bool):
        y_pred = self.clf.predict(self.X_test)
        self.confusion_matrix()
        print("Accuracy:", metrics.accuracy_score(self.y_test, y_pred))

    def confusion_matrix(self):
        y_pred = self.clf.predict(self.X_test)

        cfm = confusion_matrix(y_true=self.y_test, y_pred=y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cfm)
        disp.plot()
        plt.show()
