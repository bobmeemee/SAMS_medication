import pandas as pd


def stringVariableToInteger(var: str, classes: list):
    if var in classes:
        return classes.index(var)
    else:
        classes.append(var)
        return classes.index(var)


def columnsToIntegers(data, col_names: list):
    for column in col_names:
        classes = []
        data[column] = data[column].apply(lambda x: stringVariableToInteger(x, classes))
    return data


def dropIncomplete(data):
    for column in data:
        data.drop(data[data[column] == "#"].index, inplace=True)
    return data


def loadData(col_names):
    data = pd.read_csv("datasets/research_data_SAMS.csv", header=0, names=col_names, sep=';', index_col=False)
    return data


def countLabels(toCount):
    result = dict()
    for obj in toCount:
        if obj not in result.keys():
            result[obj] = 1
        else:
            result[obj] = result[obj] + 1
    return result
