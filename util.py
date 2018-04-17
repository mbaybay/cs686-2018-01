import pandas as pd
from sys import stderr
from sklearn.metrics import confusion_matrix


def read_arff_as_df(filepath):
    """

    :param filepath:
    :return: pandas data frame
    """
    data = pd.DataFrame()
    colnames = []
    dtypes = []
    try:
        with open(filepath) as f:
            lines = f.readlines()
            # parse header for attribute and data-type info
            for n_skip, line in enumerate(lines):
                if line.startswith('%'):
                    continue
                elif line.upper().startswith('@ATTRIBUTE'):
                    tokens = line.split()
                    # 0: @ATTRIBUTE
                    # 1: attribute name
                    # 2: attribute data type
                    colnames.append(tokens[1])
                    dtypes.append(tokens[2])
                elif line.upper().startswith('@DATA'):
                    n_skip += 1 # increment to skip @DATA line, increment again bc count started with 0
                    break

        data = pd.read_csv(filepath, names=colnames, skiprows=n_skip)

        return data

    except IOError:
        print('[ERROR] Could not read file.', file=stderr)


def print_confusion_matrix(y_true, y_pred, labels=None):
    """

    :param y_true:
    :param y_pred:
    :param labels:
    :return:
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print('-----------------------------')
    print('\tConfusion Matrix')
    print('-----------------------------')
    pred_labels = '\t'
    for label in labels:
        pred_labels += '\t' + str(label)
    print('\t\tPredicted')
    print('\t\t' + str(labels))
    print('Actual')
    for i, label in enumerate(labels):
        print(str(label) + '\t\t' + str(cm[i]))
