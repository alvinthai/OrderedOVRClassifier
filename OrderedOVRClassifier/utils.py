from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as m


def plot_thresholds(clf, X, y_true, title=None):
    '''
    Function for plotting the accuracy, precision, recall, f1, and weighted f1
    of probability predictions for a binary classifier.

    Returns the probability for the best weighted f1 score, which is a class
    imbalance average of the f1 scores for both positive and negative class.
    Weighted f1 Score is generally a useful metric when a prediction for every
    data point matters, as no classes are considered more important than the
    other.

    Parameters
    ----------
    clf: model
        Fitted classifier with a predict_proba function.
    X: array-like, shape = [n_samples, n_features]
        Data to predict with classifier.
    y_true: array-like, shape = [n_samples, ]
        Ground truth (correct) target values.
    title: str, optional
        Title for plots.

    Returns
    -------
    best: float
        Threshold that maximizes the weighted f1 score.
    '''
    probas = clf.predict_proba(X)

    thlds = np.arange(0, 1.01, .01)
    scores, scores0, scores1 = [], [], []

    if title is None:
        title = ''

    for thld in thlds:
        pred = probas[:, 1] >= thld

        p, r, f, s = m.precision_recall_fscore_support(y_true, pred,
                                                       labels=[0, 1])
        ac = m.accuracy_score(y_true, pred)
        fw = np.average(f, weights=s)

        scores.append([thld, ac, fw])
        scores0.append([thld, p[0], r[0], f[0]])
        scores1.append([thld, p[1], r[1], f[1]])

    scores = np.array(scores)
    best = np.argmax(scores[:, 2])

    cols0 = ['threshold', 'accuracy', 'f1_weighted']
    cols1 = ['threshold', 'precision_class0', 'recall_class0', 'f1_class0']
    cols2 = ['threshold', 'precision_class1', 'recall_class1', 'f1_class1']

    scores = pd.DataFrame(scores, columns=cols0)
    scores0 = pd.DataFrame(scores0, columns=cols1)
    scores1 = pd.DataFrame(scores1, columns=cols2)

    for df in [scores, scores0, scores1]:
        df = df.set_index('threshold')

        df.plot()
        plt.vlines(best/100, 0, 1, color='purple', linestyle='--',
                   label='best f1_weighted')
        plt.legend()
        plt.ylabel('score')
        plt.title(title)
        plt.show(block=False)

    print '-'*80
    print 'best_threshold:', thlds[best]
    print 'pred_cnt:', (probas[:, 1] >= thlds[best]).sum()
    print 'true_cnt:', y_true.sum()
    print
    extended_classification_report(y_true, probas[:, 1] >= thlds[best])

    return best


def extended_classification_report(y_true, y_pred):
    '''
    Extention of sklearn.metrics.classification_report. Builds a text report
    showing the main classification metrics and the total count of multiclass
    predictions per class.

    Parameters
    ----------
    y_true: array-like, shape = [n_samples, ]
        Ground truth (correct) target values.
    y_pred: array-like, shape = [n_samples, ]
        Estimated targets as returned by a classifier.
    '''
    acc = m.accuracy_score(y_true, y_pred)

    output = m.classification_report(y_true, y_pred, digits=3)
    class_labels = sorted(pd.Series(y_true).unique())

    n_pred = pd.Series(y_pred).value_counts()

    if class_labels == [False, True] and type(class_labels[0]) != np.int64:
        n_pred = np.array([n_pred[False], n_pred[True]])
    else:
        n_pred = n_pred[class_labels].values

    padding = max([15, np.ceil(max(np.log10(n_pred))) + 2])
    n_pred = np.char.array(n_pred)

    output = output.split('\n')
    output[0] += 'n_predictions'.rjust(padding)

    for i, x in enumerate(n_pred):
        output[i+2] += x.rjust(padding)

    output.extend(['', 'accuracy: {}'.format(acc), ''])

    print('\n'.join(output))
