import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from models.NearestNeighborModel import NearestNeighborModel
from options.nearest_neighbor_options import NearestNeighborOptions
from utils.utils import loadData, dropIncomplete, columnsToIntegers, countLabels


if __name__ == '__main__':
    options = NearestNeighborOptions()

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
    model = NearestNeighborModel(data, options)

    # scale model
    # model.scale_model(StandardScaler()) # comment if testing iterations

    # train model
    model.train_model()

    # test model accuracy
    model.test_model(isMultilabel=True)


