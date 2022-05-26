import numpy as np
import pandas as pd

from models.RandomForest import RandomForestModel

from options.random_forest_options import RandomForestOptions

from utils.utils import loadData, dropIncomplete, columnsToIntegers

if __name__ == '__main__':
    options = RandomForestOptions()

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
    model = RandomForestModel(data, options)

    # train model
    model.train_model()

    # test model accuracy
    model.test_model(isMultilabel=True)

