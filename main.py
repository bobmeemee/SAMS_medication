import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from models.DecisionTreeModel import DecisionTreeModel
from models.RandomForest import RandomForestModel
from options.decision_tree_options import DecisiontreeOptions
from options.random_forest_options import RandomForestOptions
from utils.utils import loadData, dropIncomplete, columnsToIntegers, countLabels


# only for rfc, maybe to other class?
def train_size_variable(start, step, stop, d, opt):
    prediction_l = list()
    test_l = list()
    for i in range(start, stop, step):
        options.set_train_size(i / 100)

        m = DecisionTreeModel(d, opt)
        m.scale_model(StandardScaler())
        m.train_model()

        out = m.count_test_split(toPrint=False)
        test_l.append(out[0])
        prediction_l.append(out[1])

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)

    ax1.plot(range(start, stop, step), prediction_l, color='tab:blue')
    ax1.plot(range(start, stop, step), test_l, color='tab:orange')
    ax1.set_title('Train vs test label amtount with variable test')
    plt.show()


if __name__ == '__main__':
    options = DecisiontreeOptions()

    # load and prepare data
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

    # train model
    model = DecisionTreeModel(data, options)

    # scale model 0-1
    scaler = StandardScaler()
    model.scale_model(scaler)

    # train on data
    model.train_model()

    # test model accuracy
    model.test_model(isMultilabel=True)
