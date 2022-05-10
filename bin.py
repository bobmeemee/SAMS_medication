import pandas as pd
import pydotplus
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from io import StringIO
from IPython.display import Image

import pydotplus
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from io import StringIO
from IPython.display import Image

from models.Model import Model
from options.random_forest_options import RandomForestOptions
from utils.utils import stringVariableToInteger, loadData, columnsToIntegers, dropIncomplete


def run_dst():
    col_names = ['age', 'sex', 'marital_status', 'occupation', 'education', 'med_prep_by', 'medication',
                 'know_reason', 'know_dosage', 'familiar_timing',
                 'take_regurarly', 'know_med', 'forget_med', 'untroubled_after_dose', 'stop_med_feel_better',
                 'stop_med_feel_worse', 'other_med_if_side_effects', 'reduce_med_no_consult', 'break_from_med',
                 'to_many_med_stop_no_consult', 'no_med_morning', 'no_med_noon', 'no_med_evening',
                 'take_only_considered_important',
                 'weekly_med_forget', 'all_columns_together']

    # https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    sams = loadData(col_names)

    # drop all rows with incomplete data, 381 entries left
    sams = dropIncomplete(sams)

    to_convert = ['sex', 'marital_status', 'occupation', 'education', 'med_prep_by']
    sams = columnsToIntegers(sams, to_convert)

    feature_cols = ['age', 'sex', 'marital_status', 'occupation', 'education', 'med_prep_by', 'medication',
                    'know_reason', 'know_dosage', 'familiar_timing',
                    'know_med', 'forget_med', 'untroubled_after_dose', 'stop_med_feel_better',
                    'stop_med_feel_worse', 'other_med_if_side_effects', 'reduce_med_no_consult', 'break_from_med',
                    'to_many_med_stop_no_consult', 'no_med_morning', 'no_med_noon', 'no_med_evening',
                    'take_only_considered_important',
                    'weekly_med_forget', 'all_columns_together']

    X = sams[feature_cols]  # Features
    y = sams['take_regurarly']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5,
                                                        random_state=1)  # 70% training and 30% test

    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(X_train)

    Standardized_X_train = scaler.transform(X_train)
    Standardized_X_test = scaler.transform(X_test)

    # Create Decision Tree classifer object
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    clf = DecisionTreeClassifier(criterion="entropy")

    # Train Decision Tree Classifer
    clf = clf.fit(Standardized_X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(Standardized_X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('sams-medication.png')
    Image(graph.create_png())


def run_rfc():
    col_names = ['age', 'sex', 'marital_status', 'occupation', 'education', 'med_prep_by', 'medication',
                 'know_reason', 'know_dosage', 'familiar_timing',
                 'take_regurarly', 'know_med', 'forget_med', 'untroubled_after_dose', 'stop_med_feel_better',
                 'stop_med_feel_worse', 'other_med_if_side_effects', 'reduce_med_no_consult', 'break_from_med',
                 'to_many_med_stop_no_consult', 'no_med_morning', 'no_med_noon', 'no_med_evening',
                 'take_only_considered_important',
                 'weekly_med_forget', 'all_columns_together']

    # https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    sams = loadData(col_names)
    sams = dropIncomplete(sams)
    text_col = ['sex', 'marital_status', 'occupation', 'education', 'med_prep_by']
    sams = columnsToIntegers(sams, text_col)

    # drop all rows with incomplete data, 381 entries left

    feature_cols = ['age', 'sex', 'marital_status', 'occupation', 'education', 'med_prep_by', 'medication',
                    'know_reason', 'know_dosage', 'familiar_timing',
                    'know_med', 'forget_med', 'untroubled_after_dose', 'stop_med_feel_better',
                    'stop_med_feel_worse', 'other_med_if_side_effects', 'reduce_med_no_consult', 'break_from_med',
                    'to_many_med_stop_no_consult', 'no_med_morning', 'no_med_noon', 'no_med_evening',
                    'take_only_considered_important',
                    'weekly_med_forget', 'all_columns_together']

    X = sams[feature_cols]  # Features
    y = sams['take_regurarly']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                        random_state=1)  # 70% training and 30% test

    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(X_train)

    Standardized_X_train = scaler.transform(X_train)
    Standardized_X_test = scaler.transform(X_test)

    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0,  # randomness of bootstrapping control
                               shuffle=False)

    clf = RandomForestClassifier(max_depth=2, random_state=0)

    clf = clf.fit(Standardized_X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(Standardized_X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


run_dst()
