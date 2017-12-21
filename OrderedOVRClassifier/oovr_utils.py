from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as m


class OOVR_Model(object):
    '''
    Generic class for storing attributes for OrderedOVRClassifier.

    Parameters
    ----------
    attributes: dict
        Dictionary of attributes to be stored for later loading.
    '''
    def __init__(self, attributes):
        for k, v in attributes.iteritems():
            self.__setattr__(k, v)


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


def plot_oovr_dependencies(oovr, ovr_val, X, y, comp_vals=None):
    '''
    Evaluates the effect of changing the threshold of an ordered OVR
    classifier against other classes with respect to accuracy, precision,
    recall, and f1 metrics.

    Parameters
    ----------
    oovr: Trained OrderedOVRClassifier model.
    ovr_val: str, int, or float
        Class label to evaluate metrics against other classes.
    X: array-like, shape = [n_samples, n_features]
        Data used for predictions.
    y: array-like, shape = [n_samples, ], optional
        True labels for X. If not provided and X is a DataFrame, will
        extract y column from X with the provided self.target value.
    comp_vals: list of str, optional
        List of classes to compare against the trained classifier for
        ovr_val. If None, all other classes will be compared against the
        ovr_val class.
    '''
    if oovr.pipeline[-1][0] != 'final':
        raise AssertionError("Error: No final model attached.")

    if ovr_val not in oovr.ovr_vals:
        msg = "Error: Can't plot dependencies for a non-ordered classifier."
        raise AssertionError(msg)

    if comp_vals is None:
        comp_vals = list(set(oovr._le.classes_) - set([ovr_val]))

    # Find indexes for column later subslicing
    comp_vals = sorted(comp_vals)
    comp_idxs = oovr._le.transform(comp_vals).tolist()
    ovr_idx = oovr._le.transform([ovr_val])[0]
    ovr_pipe_idx = oovr.ovr_vals.index(ovr_val)
    earlier_pipe_idx = np.arange(len(oovr.pipeline))[:ovr_pipe_idx]

    # Create mask, make predictions for OVR classifier and reverse pipeline
    mask = np.zeros(len(y)).astype(bool)
    proba_ovr = oovr.pipeline[ovr_pipe_idx][1].predict_proba(X)[:, 1]
    pred_partial = oovr.predict(X, start=ovr_pipe_idx+1)

    if ovr_idx > 0:
        pred_final = oovr.predict(X, start=0)

    # Do not do comparisons against earlier pipeline steps (if applicable)
    # Also masks out predictions from earlier pipeline steps (if applicable)
    for i in earlier_pipe_idx:
        lbl = oovr.pipeline[i][0]

        if lbl in comp_vals:
            comp_vals.remove(l)
            comp_idxs.remove(oovr._le.transform([l]))

            print('Will not evaluate partial classification dependencies for '
                  '"{0} vs. {1}" because {1} is classified at an earlier step '
                  'than {0}.'.format(ovr_val, lbl))

        mask = np.logical_or(mask, pred_final == lbl)

    cols_slice = np.r_[ovr_idx, comp_idxs]

    # Repeat predictions over all thresholds with 0.01 interval spacing
    pred_partial = np.repeat(pred_partial, 100)
    proba_ovr = np.repeat(proba_ovr, 100)
    thlds = np.tile(np.arange(0, 1.00, 0.01), len(y))

    # Set predictions above or equal to threshold
    pred_partial[proba_ovr >= thlds] = ovr_val
    pred_partial = pred_partial.reshape(-1, 100).T

    # ============================================================
    def accuracy_compute(y_pred):
        # note y and mask are variables not local to myfunc
        accs = m.accuracy_score(y[~mask], y_pred[~mask])
        return accs

    def classification_compute(y_pred):
        # note y and mask are variables not local to myfunc
        prf = m.precision_recall_fscore_support(y[~mask], y_pred[~mask],
                                                pos_label=None)[0:3]
        prf = np.ravel(np.vstack(prf)[:, cols_slice].T)
        return prf
    # ============================================================

    # Calculate accuracy scores across thresholds
    accs = np.apply_along_axis(accuracy_compute, 1, pred_partial)

    # Plot accuracy as a function of threshold for OVR classifier
    pd.DataFrame(accs, index=np.arange(0, 1.00, 0.01),
                 columns=['accuracy']).plot()

    plt.title('{} vs Threshold'.format(ovr_val))
    plt.ylabel('accuracy')
    plt.xlabel('thresholds')
    plt.xticks(np.arange(0, 1.01, 0.1))
    plt.legend(loc=2, bbox_to_anchor=(1, 1))
    plt.grid(linestyle=':')

    plt.show(block=False)

    # Calculate precison, recall, f1 scores across thresholds
    prf = np.apply_along_axis(classification_compute, 1, pred_partial)
    cols = [['precision_' + str(l), 'recall_' + str(l), 'f1_' + str(l)]
            for l in oovr._le.inverse_transform(cols_slice)]
    cols = np.ravel(cols)

    # Plot precision, recall, and f1 as a function of threshold for ovr_val
    # vs comparison classes
    for i, comp_val in enumerate(comp_vals, 1):
        subslice = np.r_[np.arange(3), np.arange(3*i, 3*i + 3)]
        pd.DataFrame(prf[:, subslice], index=np.arange(0, 1.00, 0.01),
                     columns=cols[subslice]).plot()

        plt.title('{} vs {}'.format(ovr_val, comp_val))
        plt.ylabel('scores')
        plt.xlabel('thresholds')
        plt.yticks(np.arange(0, 1.01, 0.1))
        plt.xticks(np.arange(0, 1.01, 0.1))
        plt.legend(loc=2, bbox_to_anchor=(1, 1), ncol=2)
        plt.grid(linestyle=':')

        plt.show(block=False)


def plot_thresholds(clf, X, y_true, beta=1.0, title=None):
    '''
    Function for plotting the accuracy, precision, recall, fscore, and weighted
    fscore of probability predictions for a binary classifier.

    Returns the probability for the best weighted fscore, which is a class
    imbalance average of the fscores for both positive and negative class.
    Weighted fscore is generally a useful metric when a prediction for every
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
        Threshold that maximizes the weighted fscore.
    scores_at_best: ([1, 3], [1, 3]) tuple
        A tuple with the class [precision, recall, fscore] scores at the best
        threshold. 1st element contains scores for the False class and 2nd
        element contains scores for the True class.
    '''
    probas = clf.predict_proba(X)

    thlds = np.arange(0, 1, 0.01)
    scores, scores0, scores1 = [], [], []

    if title is None:
        title = ''

    for thld in thlds:
        pred = probas[:, 1] >= thld

        p, r, f, s = m.precision_recall_fscore_support(y_true, pred, beta=beta,
                                                       labels=[0, 1])
        ac = m.accuracy_score(y_true, pred)
        fw = np.average(f, weights=s)

        scores.append([thld, ac, fw])
        scores0.append([thld, p[0], r[0], f[0]])
        scores1.append([thld, p[1], r[1], f[1]])

    scores = np.array(scores)
    best = np.argmax(scores[:, 2])

    if beta == 1.0:
        flabel = 'f1'
    else:
        flabel = 'fbeta'

    cols = ['threshold', 'accuracy', '{}_weighted'.format(flabel)]
    cols0 = ['threshold', 'precision_False', 'recall_False',
             '{}_True'.format(flabel)]
    cols1 = ['threshold', 'precision_True', 'recall_True',
             '{}_True'.format(flabel)]

    scores = pd.DataFrame(scores, columns=cols)
    scores0 = pd.DataFrame(scores0, columns=cols0)
    scores1 = pd.DataFrame(scores1, columns=cols1)
    scores_at_best = (scores0.iloc[best, 1:].values,
                      scores1.iloc[best, 1:].values)

    for df in [scores, scores0, scores1]:
        df = df.set_index('threshold')

        df.plot()
        plt.vlines(best/100, 0, 1, color='purple', linestyle='--',
                   label='best {}_weighted'.format(flabel))

        plt.title(title)
        plt.ylabel('score')
        plt.yticks(np.arange(0, 1.01, 0.1))
        plt.xticks(np.arange(0, 1.01, 0.1))
        plt.legend(loc=2, bbox_to_anchor=(1, 1))
        plt.grid(linestyle=':')

        plt.show(block=False)

    print '-'*80
    print 'best_threshold:', thlds[best]
    print 'pred_cnt:', (probas[:, 1] >= thlds[best]).sum()
    print 'true_cnt:', y_true.sum()
    print
    extended_classification_report(y_true, probas[:, 1] >= thlds[best])

    return best, scores_at_best
