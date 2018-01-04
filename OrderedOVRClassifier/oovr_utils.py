from __future__ import division, print_function
from matplotlib.ticker import FormatStrFormatter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as m

try:
    from modified_packages.skater_modified import PartialDependence
    from skater.core.explanations import Interpretation
    from skater.core.global_interpretation.feature_importance import FeatureImportance
    from skater.model import InMemoryModel
    skater_loaded = True
except ImportError:
    skater_loaded = False

try:
    from tqdm import tqdm
    tqdm_loaded = True
except ImportError:
    tqdm_loaded = False


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


class PipelineES(Pipeline):
    '''
    Modified Pipeline class that allows transformed dataset to be passed into
    the 'eval_set' parameter when fitting LGBMClassifier or XGBClassifier.
    '''
    def _transform(self, X):
        Xt = X
        for name, transform in self.steps:
            if hasattr(transform, 'transform'):
                Xt = transform.transform(Xt)
        return Xt

    def fit(self, X, y=None, eval_idx=None, **fit_params):
        """Fit the model

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.
        If eval_idx is passed, splits the transformed data into train and eval
        sets prior to fitting final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        eval_idx: tuple --> (list, list), default=None
            Tuple of indexes for splitting X into train and eval datasets.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : Pipeline
            This estimator
        """
        if eval_idx is not None:
            _, fit_params = self._fit(indexer(X, eval_idx[0][0]),
                                      indexer(y, eval_idx[0][0]), **fit_params)
            Xt = self._transform(X)
        else:
            Xt, fit_params = self._fit(X, y, **fit_params)

        if self._final_estimator is not None:
            early_stop_models = ['lightgbm.sklearn', 'xgboost.sklearn']

            if eval_idx is not None:
                es = [(indexer(Xt, eval_idx[0][1]),
                       indexer(y, eval_idx[0][1]))]
                Xt = indexer(Xt, eval_idx[0][0])
                y = indexer(y, eval_idx[0][0])
            else:
                es = None

            if self._final_estimator.__module__ in early_stop_models:
                self._final_estimator.fit(Xt, y, eval_set=es, **fit_params)
            else:
                self._final_estimator.fit(Xt, y, **fit_params)

        return self


class UniformClassifier(BaseEstimator, ClassifierMixin):
    '''
    Dumb classifier that predicts the same value for all predictions.

    Parameters
    ----------
    val: str or numeric
        Value to return in predictions.
    '''
    def __init__(self, val):
        self.val = val

    def fit(self, X=None, y=None):
        return self

    def predict(self, X):
        return np.full(len(X), self.val)

    def predict_proba(self, X):
        return np.ones(len(X)).reshape(-1, 1)


def check_eval_metric(clf, fit_params, eval_X, eval_y, prefix=None):
    '''
    A function for ensuring default metrics for early stopping
    evaluation are compatible with xgboost and lightgbm. Also adds
    eval_set into fit parameters for xgboost and lightgbm.

    The default evaluation metric for binary classification is accuracy
    and the default evaluation metric for multiclass classification is
    multiclass logloss.

    Parameters
    ----------
    clf: model
        Unfitted model. Used to check the type of model being fitted.

    fit_params: dict
        Fit parameters to pass into clf.

    eval_X: array-like, shape = [n_samples, n_features]
        X input evaluation data for early stopping.

    eval_y: array-like, shape = [n_samples, ]
        True classification values to score early stopping.

    prefix: str, optional
        String prefix for parameter keys.

    Returns
    -------
    fit_params: dict
        Early stopping compatible fit parameters to pass into clf.
    '''
    early_stop_berror = {'xgboost.sklearn': 'error',
                         'lightgbm.sklearn': 'binary_error'}
    early_stop_mlog = {'xgboost.sklearn': 'mlogloss',
                       'lightgbm.sklearn': 'multi_logloss'}
    berror = early_stop_berror.values()
    mlog = early_stop_mlog.values()
    module = clf.__module__

    if module == 'sklearn.model_selection._search':
        module = clf.estimator.__module__

    if module in early_stop_berror:
        if prefix is None:
            prefix = ''
            fit_params['eval_set'] = [(eval_X, eval_y)]

        binary = len(np.unique(eval_y)) == 2

        if prefix + 'early_stopping_rounds' not in fit_params:
            fit_params[prefix + 'early_stopping_rounds'] = 10

        if prefix + 'eval_metric' not in fit_params:
            if binary:
                fit_params[prefix + 'eval_metric'] = early_stop_berror[module]
            else:
                fit_params[prefix + 'eval_metric'] = early_stop_mlog[module]
        elif binary:
            if fit_params[prefix + 'eval_metric'] in mlog:
                fit_params[prefix + 'eval_metric'] = early_stop_berror[module]
        elif fit_params[prefix + 'eval_metric'] in berror:
            fit_params[prefix + 'eval_metric'] = early_stop_mlog[module]

    return fit_params


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


def indexer(arr, indexes):
    '''
    Performs an interger indexing operation based on the data type of <arr>.

    Parameters
    ----------
    arr: array-like, shape = [n_samples, n_features]
        Pandas or Numpy object to index.

    indexes: array-like, shape = [n, ]
        List of interger indexes to slice.

    Returns
    -------
    arr: array-like, shape = [n, n_features]
        Pandas or Numpy object indexed to rows specified in <indexes>.
    '''
    if arr.__class__ in [pd.core.frame.DataFrame, pd.core.series.Series]:
        return arr.iloc[indexes]
    else:
        return arr[indexes]


def plot_2d_partial_dependence(oovr, X, col, col_names=None,
                               grid_resolution=100, grid_range=(.05, 0.95),
                               n_jobs=-1, n_samples=1000, progressbar=True):
    '''
    Wrapper function for calling the plot_partial_dependence function from
    skater, which estimates the partial dependence of a column based on a
    random sample of 1000 data points. To calculate partial dependencies the
    following procedure is executed:

    1. Pick a range of values (decided by the grid_resolution and grid_range
       parameters) to calculate partial dependency for.
    2. Loop over the values, one at a time, repeating steps 3-5 each time.
    3. Replace the entire column corresponding to the variable of interest with
       the current value that is being cycled over.
    4. Use the model to predict the probabilities.
    5. The (value, average_probability) becomes an (x, y) pair of the partial
       dependence plot.

    Parameters
    ----------
    oovr: OrderedOVRClassifier
        Trained OrderedOVRClassifier model.

    X: array-like, shape = [n_samples, n_features]
        Input data used for training or evaluating the fitted model.

    col: str
        Label for the feature to compute partial dependence for.

    col_names: list, optional
        Names to call features.

    grid_resolution: int, optional, default: 100
        How many unique values to include in the grid. If the percentile range
        is 5% to 95%, then that range will be cut into <grid_resolution>
        equally size bins.

    grid_range: (float, float) tuple, optional, default: (.05, 0.95)
        The percentile extrama to consider. 2 element tuple, increasing,
        bounded between 0 and 1.

    n_jobs: int, optional, default: -1
        The number of CPUs to use to compute the partial dependence. -1 means
        'all CPUs' (default).

    n_samples: int, optional, default: 1000
        How many samples to use when computing partial dependence.

    progressbar: bool, optional, default: True
        Whether to display progress. This affects which function we use to
        multipool the function execution, where including the progress bar
        results in 10-20% slowdowns.
    '''
    if not skater_loaded:
        raise RuntimeError("Skater is required but not installed. Please "
                           "install skater with 'pip install skater' to use "
                           "this function.")

    target_names = ['predicted_' + str(x) for x in oovr._le.classes_]

    interpreter = Interpretation(X, feature_names=col_names)
    interpreter.logger.handlers = [logging.StreamHandler()]

    pdep = PartialDependence(interpreter)
    pyint_model = InMemoryModel(oovr.predict_proba, target_names=target_names,
                                examples=X)

    fig = pdep.plot_partial_dependence([col], pyint_model, with_variance=False,
                                       figsize=(6, 4),
                                       grid_resolution=grid_resolution,
                                       grid_range=grid_range,
                                       n_jobs=n_jobs, n_samples=n_samples,
                                       progressbar=progressbar)

    for i, f in enumerate(fig[0]):
        if f.__module__ == 'matplotlib.axes._subplots':
            f.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            f.set_title('Partial Dependence Plot')

    plt.show()


def plot_feature_importance(oovr, X, y, col_names=None, filter_class=None,
                            n_jobs=-1, n_samples=5000, progressbar=True):
    '''
    Wrapper function for calling the plot_feature_importance function from
    skater, which estimates the feature importance of all columns based on a
    random sample of 5000 data points. To calculate feature importance the
    following procedure is executed:

    1. Calculate the original probability predictions for each class.
    2. Loop over the columns, one at a time, repeating steps 3-5 each time.
    3. Replace the entire column corresponding to the variable of interest with
       replacement values randomly sampled from the column of interest.
    4. Use the model to predict the probabilities.
    5. The (column, average_probability_difference) becomes an (x, y) pair of
       the feature importance plot.
    6. Normalize the average_probability_difference so the sum equals 1.

    Parameters
    ----------
    oovr: OrderedOVRClassifier
        Trained OrderedOVRClassifier model.

    X: array-like, shape = [n_samples, n_features]
        Input data used for training or evaluating the fitted model.

    y: array-like, shape = [n_samples, ]
        True labels for X.

    col_names: list, optional
        Names to call features.

    filter_class: str or numeric, optional
        If specified, the feature importances will only be calculated for y
        data points matching class specified for filter_class.

    n_jobs: int, optional, default: -1
        The number of CPUs to use to compute the feature importances. -1 means
        'all CPUs' (default).

    n_samples: int, optional, default: 5000
        How many samples to use when computing importance.

    progressbar: bool, optional, default: True
        Whether to display progress. This affects which function we use to
        multipool the function execution, where including the progress bar
        results in 10-20% slowdowns.
    '''
    if not skater_loaded:
        raise RuntimeError("Skater is required but not installed. Please "
                           "install skater with 'pip install skater' to use "
                           "this function.")

    target_names = oovr._le.classes_

    if filter_class is not None:
        title = 'Feature Importances: Class = {}'.format(filter_class)
        filter_class = list(filter_class)
    else:
        title = 'Feature Importances'

    interpreter = Interpretation(X, training_labels=y, feature_names=col_names)
    interpreter.logger.handlers = [logging.StreamHandler()]

    feat = FeatureImportance(interpreter)
    pyint_model = InMemoryModel(oovr.predict_proba, target_names=target_names,
                                examples=X)

    fig, ax = feat.plot_feature_importance(pyint_model,
                                           filter_classes=filter_class,
                                           n_jobs=n_jobs,
                                           n_samples=n_samples,
                                           progressbar=progressbar)
    fig.set_size_inches(18.5, max(ax.get_ylim()[1] / 4, 10.5))
    ax.set_title(title)

    plt.show()


def plot_oovr_dependencies(oovr, ovr_val, X, y, comp_vals=None):
    '''
    Evaluates the effect of changing the threshold of an ordered OVR
    classifier against other classes with respect to accuracy, precision,
    recall, and f1 metrics.

    Parameters
    ----------
    oovr: OrderedOVRClassifier
        Trained OrderedOVRClassifier model.

    ovr_val: str, int, or float
        Class label to evaluate metrics against other classes.

    X: array-like, shape = [n_samples, n_features]
        Data used for predictions.

    y: array-like, shape = [n_samples, ]
        True labels for X.

    comp_vals: list of str, optional
        List of classes to compare against the trained classifier for
        ovr_val. If None, all other classes will be compared against the
        ovr_val class.
    '''
    if oovr.pipeline[-1][0] != 'final':
        raise RuntimeError("Error: No final model attached.")

    if ovr_val not in oovr.ovr_vals:
        msg = "Error: Can't plot dependencies for a non-ordered classifier."
        raise RuntimeError(msg)

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
    def accuracy_compute(y_pred, pbar=None):
        # note y and mask are variables not local to myfunc
        accs = m.accuracy_score(y[~mask], y_pred[~mask])
        if pbar is not None:
            pbar.update()
        return accs

    def classification_compute(y_pred, pbar=None):
        # note y and mask are variables not local to myfunc
        prf = m.precision_recall_fscore_support(y[~mask], y_pred[~mask],
                                                warn_for=(),
                                                pos_label=None)[0:3]
        prf = np.ravel(np.vstack(prf)[:, cols_slice].T)
        if pbar is not None:
            pbar.update()
        return prf
    # ============================================================

    # Calculate accuracy scores across thresholds
    if tqdm_loaded:
        pbar1 = tqdm(total=100, desc='Calculating accuracy')
        accs = np.apply_along_axis(accuracy_compute, 1, pred_partial,
                                   pbar=pbar1)
        pbar1.close()
    else:
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
    if tqdm_loaded:
        pbar2 = tqdm(total=100, desc='Calculating precision, recall, and f1')
        prf = np.apply_along_axis(classification_compute, 1, pred_partial,
                                  pbar=pbar2)
        pbar2.close()
    else:
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

    beta: float, optional, default: 1.0
        The strength of recall versus precision in the F-score.

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
                                                       labels=[0, 1],
                                                       warn_for=())
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

    print('-'*80)
    print('best_threshold:', thlds[best])
    print('pred_cnt:', (probas[:, 1] >= thlds[best]).sum())
    print('true_cnt:', y_true.sum(), '\n')

    extended_classification_report(y_true, probas[:, 1] >= thlds[best])

    return best, scores_at_best
