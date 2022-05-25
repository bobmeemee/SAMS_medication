import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from models.DecisionTreeModel import DecisionTreeModel
from models.LogisticRegressionModel import LogisticRegressionModel
from models.NearestNeighborModel import NearestNeighborModel
from models.RandomForest import RandomForestModel
from options.logistic_regression_options import LogisticRegressionOptions

from options.random_forest_options import RandomForestOptions

from utils.utils import loadData, dropIncomplete, columnsToIntegers, countLabels

if __name__ == '__main__':
    options = LogisticRegressionOptions()

    # load and preprocess
    data = loadData(options.col_names)
    data = dropIncomplete(data)
    data = columnsToIntegers(data, options.notIntegerColumns)

    # categorize last column according to SAMS research
    data['total_score_cat'] = pd.cut(
        x=data['total_score'],
        bins=[-1, 0, 11, np.inf],
        labels=[0, 1, 2],
    )

    print(data[options.target_col].value_counts())

    # build model
    model = LogisticRegressionModel(data, options)

    # train model
    model.train_model()

    # test model accuracy
    model.test_model(isMultilabel=True)


