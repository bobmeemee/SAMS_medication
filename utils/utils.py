import pandas as pd


# convert all strings from a column to integers in classes
def stringVariableToInteger(var: str, classes: list):
    if var in classes:
        return classes.index(var)
    else:
        classes.append(var)
        return classes.index(var)

# takes data and a list of columns as input
# returns chosen columns converted to integer data
def columnsToIntegers(data, col_names: list):
    for column in col_names:
        classes = []
        data[column] = data[column].apply(lambda x: stringVariableToInteger(x, classes))
    return data


# drops columns from data that is not complete
def dropIncomplete(data):
    for column in data:
        data.drop(data[data[column] == "#"].index, inplace=True)
    return data

# load dataset to pandas file with chosen names col_names
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
