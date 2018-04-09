from svm import SVM
import pandas as pd
from sklearn.model_selection import train_test_split

def accuracy(labels, hypotheses):
    count = 0.0
    correct = 0.0

    for l, h in zip(labels, hypotheses):
        count += 1.0
        if l == h:
            correct += 1.0
    return correct / count


def print_confusion_matrix(labels, hypotheses):
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    count = 1.0
    for l, h in zip(labels, hypotheses):
        count += 1.0
        if l == 1 and h == 1:
            tp += 1.0
        elif l == 1 and h == 0:
            tp += 1.0
        elif l == 0 and h == 0:
            tn += 1.0
        else:
            fn += 1
    print('-----------------------------')
    print('\tConfusion Matrix')
    print('-----------------------------')
    print('\t\tPredicted')
    print('\tActual\tNO\tYES')
    print('-----------------------------')
    print('\tNO\t', tn, '\t', fp)
    print('-----------------------------')
    print('\tYES\t', fn, '\t', tp)
    print('-----------------------------')


def get_data(filename):
    data = pd.read_csv(filename, sep=",\s", header=None)
    x = data.iloc[:, :-1]
    y = data.iloc[:, 2]
    return x.as_matrix(), y.as_matrix()


if __name__ == '__main__':
    x, y = get_data("../../data/linearly_separable.csv")
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)
    clf = SVM()
    weights = clf.fit(train_x, train_y)
    print("weights: {}".format(weights))
    pred = clf.predict(test_x)
    print("pred: {}".format(pred))
    print('Accuracy:', accuracy(test_y, pred))

# verify_x, verify_y = get_data('verify.txt')
# hypotheses = clf.predict(verify_x)


#
# print_confusion_matrix(verify_y, hypotheses)

