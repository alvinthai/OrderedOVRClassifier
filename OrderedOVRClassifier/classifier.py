"""
OrderedOVRClassifier
"""

# Author: Alvin Thai <alvinthai@gmail.com>

from __future__ import division
import copy
import datetime
import json
import numpy as np
import oovr_utils as u
import pandas as pd
import sklearn.metrics as m
import warnings

from collections import defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class OrderedOVRClassifier(BaseEstimator, ClassifierMixin):
    '''
    A custom scikit-learn module for approaching multi-classification with an
    Ordered One-Vs-Rest Modeling approach. Ordered One-Vs-Rest Classification
    performs a series of One-Vs-Rest Classifications where negative results
    are moved into subsequent training with previous classifications filtered
    out. The advantages of using Ordered One-Vs-Rest Classification include:

    1. Reducing training time. Most multi-classification predictive models use
       One-Vs-Rest approaches and the predicted outcome is simply the highest
       predicted (normalized) probability among all One-Vs-Rest trained models.
       Performing One-Vs-Rest Classification in an ordered manner can reduce
       training time (by virtue of having less classes to make predictions on)
       for later modeling steps that may require heavy optimization.

    2. Screening out classes that can be predicted with high confidence. If a
       highly predictive class exists among the multi-classification labels,
       it could be advantageous to remove the highly predictive class from
       future steps to allow optimization on a smaller dataset where
       differences among the remaining classes become more evident.

    3. Evaluating the thresholds of a binary classifier against classification
       metrics of all downstream predictive models. With Ordered One-Vs-Rest
       Classification, the binary outcome from an Ordered One-Vs-Rest model can
       be optimized to achieve an ideal mix of accuracy / precision / recall
       scores among each predictive class. Call the plot_oovr_dependencies
       function on a trained final model to execute these evaluations.

    The API for OrderedOVRClassifier is designed to be user-friendly with
    pandas, numpy, and scikit-learn. There is built in functionality to support
    easy handling for early stopping on the sklearn wrapper for XGBoost and
    LightGBM. If working with DataFrames, a fit model with early stopping can
    be called using a command as simple as:

    > OrderedOVRClassifier(target='label').fit(X=train_df, eval_set=eval_df)

    OrderedOVRClassifier also runs custom evaluation functions to diagnose
    and/or plot the predictive performance of the classification after training
    each model. There is also a grid search wrapper built into the API for
    hyper-parameter tuning against classification-subsetted datasets.

    Parameters
    ----------
    target: str
        Label for target variable in pandas DataFrame. If provided, all future
        future inputs with an X DataFrame do not require an accompanying y
        input as y will be extracted from the X DataFrame; however, the target
        column must be included in the X DataFrame for all fitting steps if the
        target parameter is provided.

    ovr_vals: list
        List of target values (and ordering) to perform ordered one-vs-rest.

    model_dict: dict of models
        Dictionary of models to perform ordered one-vs-rest, dict should
        include a model for each value in ovr_vals, and if
        train_final_model=True, a model specified for 'final'.

        i.e. model_dict = { value1: LogisticRegression(),
                            value2: RandomForestClassifier(),
                           'final': XGBClassifier() }

    model_fit_params: dict of dict
        Additional parameters (inputted as a dict) to pass to the fit step of
        the models specified in model_dict.

        i.e. model_fit_params = { value1: {'sample_weight': None},
                                  value2: {'sample_weight': None},
                                 'final': {'verbose': False} }

    fbeta_weight: float, default: 1.0
        The strength of recall versus precision in the F-score.

    train_final_model: bool, default: True
        Whether to train a final model to the remaining data after OVR fits.

    train_final_only: bool, default: False
        Whether to ignore OVR modeling and to train the final model only.

    Attributes
    ----------
    eval_mask: (boolean) array-like, shape = [n_samples, ]
        An array equal to the n_samples of the evaluation set, if included by
        the user for early stopping or out-of-training evaluation, when fitting
        OrderedOVRClassifier. Used to filter out classes from previously
        trained OVR models from future modeling steps. Note that if an OVR
        model has been attached to the pipeline, the same dataset used to
        evaluate the first OVR model must stay the same to evaluate future
        OrderedOVRClassifier pipeline steps.

    input_cols: list of str, shape = [n_features]
        List of columns saved when initially fitting OrderedOVRClassifier.
        Column headers for future X and eval_set inputs (if trained using
        DataFrame) must match self.input_cols for training or prediction to
        proceed.

    mask: (boolean) array-like, shape = [n_samples, ]
        An array equal to the n_samples of the training set. Used to filter out
        classes from previously trained OVR models from future modeling steps.
        Note that if an OVR model has been attached to the pipeline, the same
        dataset used to train the first OVR model must be used to train future
        OrderedOVRClassifier pipeline steps.

    pipeline: list of (str, model) tuples
        A list of prediction steps to be performed by OrderedOVRClassifier.
        Positive classification from earlier steps in the pipe will be the
        final outputted prediction. Pipeline can be appended with additional
        models using the attach_model function.

    rest_precision: list of float
        A list of precision scores for the rest class in OVR classification.
        Used to guesstimate probabilities for predict_proba.

    thresholds: list of float
        A list of thresholds for positive classification for bianry OVR models
        in the OrderedOVRClassifier pipeline.
    '''
    def __init__(self, target=None, ovr_vals=None, model_dict=None,
                 model_fit_params=None, fbeta_weight=1.0,
                 train_final_model=True, train_final_only=False):
        self._le = LabelEncoder()
        self.eval_mask = None
        self.input_cols = None
        self.mask = None
        self.pipeline = []
        self.rest_precision = []
        self.thresholds = []

        self.target = target
        self.ovr_vals = ovr_vals
        self.model_dict = model_dict
        self.model_fit_params = model_fit_params
        self.fbeta_weight = fbeta_weight
        self.train_final_model = train_final_model
        self.train_final_only = train_final_only

        # Set default values when no inputs specified
        if self.ovr_vals is None:
            self.ovr_vals = []

        if self.model_dict is None:
            self.model_dict = {}

        if 'final' not in self.model_dict:
            for _ in xrange(1):
                try:  # default model: xgboost if installed
                    from xgboost import XGBClassifier
                    self.model_dict['final'] = XGBClassifier()
                    break

                except ImportError:
                    pass

                try:  # if xgboost not found, next default: lightgbm
                    from lightgbm import LGBMClassifier
                    self.model_dict['final'] = LGBMClassifier(n_estimators=100)
                    break

                except ImportError:  # gradientboosting in absense of above
                    from sklearn.ensemble import GradientBoostingClassifier
                    self.model_dict['final'] = GradientBoostingClassifier()

        if self.model_fit_params is None:
            self.model_fit_params = defaultdict(dict)

        for k in self.model_dict:
            if k not in self.model_fit_params:
                self.model_fit_params[k] = {}

    def __updateattr__(self, attributes):
        '''
        Updates attributes of OrderedOVRClassifier from a dictionary.
        '''
        appends = ['model', 'ovr_val', 'rest_precision', 'thresholds']
        model, ovr_val = None, 'final'

        for k, v in attributes.iteritems():
            if k in appends:
                if k == 'model':
                    model = v
                elif k == 'ovr_val':
                    ovr_val = v
                else:
                    self.__getattribute__(k).append(v)
            else:
                self.__setattr__(k, v)

        if model is not None:
            self.__getattribute__('pipeline').append((ovr_val, model))

        if ovr_val != 'final':
            self.__getattribute__('ovr_vals').append(ovr_val)

    def _encode_y(self, y, eval_y=None, lgbm_grid=False):
        '''
        Encodes y values with LabelEncoder to allow classification on string
        or numerical values.

        Parameters
        ----------
        y: array-like, shape = [n_samples, ]
            y values from the training set to encode.

        eval_y: array-like, shape = [n_samples, ], optional
            y values from the evaluation set to encode.

        lgmb_grid: boolean, default: False
            For grid search, LGBM requires input to be between [0, n_class),
            which may not be the case for OrderedOVRClassifier when classes
            are masked out. Setting this value to true ensures that a new
            LabelEncoder is always used.

        Returns
        ------
        y: array-like, shape = [n_samples, ]
            y values from the training set after LabelEncoder transform.

        eval_y: array-like, shape = [n_samples, ] or None
            y values from the evaluation set after LabelEncoder transform.
            Returns None if no eval_y is provided as input.

        enc: LabelEncoder
            LabelEncoder object for transforming the y values.
        '''
        if hasattr(self._le, 'classes_') and not lgbm_grid:
            enc = self._le
        else:
            enc = LabelEncoder().fit(y)

        y = enc.transform(y)

        if eval_y is not None:
            eval_y = enc.transform(eval_y)

        return y, eval_y, enc

    def _eval_set(self, eval_set, drop_cols):
        '''
        Cleans up eval_set into proper format for xgboost and lightgbm. If
        eval_set is a DataFrame, unpacks it into (X, y) pair. Aside from
        xgboost/lightgbm, eval_set is also used to evaluate trained models on
        unseen data and validate grid searches.

        Parameters
        ----------
        eval_set: DataFrame or list of (X, y) pair
            Dataset to use as validation set for early-stopping and/or scoring
            trained models.

        drop_cols: list of str
            Columns to drop from DataFrame prior to modeling.

        Returns
        -------
        eval_set: list of (X, y) pair
            Cleaned up eval_set for further usage in OrderedOVRClassifier.
        '''
        if eval_set is None:
            return None

        if eval_set.__class__ == pd.DataFrame:  # eval_set is a DataFrame
            eval_X = eval_set.drop(drop_cols, axis=1)
            eval_y = eval_set[self.target]
            eval_set = [(eval_X, eval_y)]

        elif eval_set[0].__class__ == tuple:  # eval_set is a (X, y) tuple
            if len(eval_set[0]) != 2:
                raise AssertionError('Invalid shape for eval_set')

            if eval_set[0][0].__class__ == pd.DataFrame:
                eval_X = eval_set[0][0].drop(drop_cols, axis=1)
                eval_set = [(eval_X, pd.Series(eval_set[0][1]))]

        else:  # bad input for eval_set
            raise TypeError('Invalid input for eval_set')

        assert len(eval_set[0][0]) == len(eval_set[0][1])

        if eval_set[0][0].__class__ == pd.DataFrame:
            if not all(self.input_cols == eval_X.columns):
                raise AssertionError("Incompatible columns! Please check "
                                     "if columns in X dataframe of "
                                     "eval_set matches self.input_cols.")

        return eval_set

    def _fit_final_model(self, X, y, eval_set, attach, model=None):
        '''
        Utility function for fitting final model or testing a model against a
        classification-masked X dataset.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Training input data.
        y: array-like, shape = [n_samples, ]
            True classification value for training data.
        eval_set: list of (X, y) pair
            Dataset to use as validation set for early-stopping and/or scoring
            trained models.
        attach: boolean
            Whether to attach the fitted model to the OrderedOVRClassifier
            pipeline.
        model: model, optional
            A model to test against X dataset with masked classifications.

        Returns
        -------
        clf: model
            Fitted model trained against classification-masked X dataset.
        '''
        Xm, ym, eval_X, eval_y = self._mask_datasets(X, y, eval_set)
        ym, eval_y, enc = self._encode_y(ym, eval_y)
        clf, fit_params = self._get_model('final', eval_X, eval_y, model)

        clf = self._fit_model(clf, Xm, ym, 'final', fit_params)

        if eval_set is not None:
            y_true = eval_y
            y_pred = clf.predict(eval_X)
        else:
            y_true = ym
            y_pred = clf.predict(Xm)

        y_true = enc.inverse_transform(y_true)
        y_pred = enc.inverse_transform(y_pred)

        if not self.train_final_only:
            print
            print '-'*80
            u.extended_classification_report(y_true, y_pred)

        if attach:
            self.pipeline.append(('final', clf))

            if not self.train_final_only:
                print '-'*80
                print 'finished fit for remaining classes'
                print '-'*80
                print
        else:
            model_attr = {'model': clf, 'enc': enc}
            return u.OOVR_Model(model_attr)

    def _fit_model(self, clf, X, y, key, fit_params):
        '''
        Wrapper for training a general model with additional fit parameters.

        Parameters
        ----------
        clf: model
            Unfitted model to train.
        X: array-like, shape = [n_samples, n_features]
            Training input data.
        y: array-like, shape = [n_samples, ]
            True classification value for training data.
        key: str
            Label from model_fit_params dict where fit_params was fetched.
        fit_params: dict
            Key-value pairs of optional arguments to pass into model fit
            function.

        Returns
        -------
        clf: model
            Fitted model trained against X dataset.
        '''
        if len(fit_params) > 0:
            try:
                clf.fit(X, y, **fit_params)
            except TypeError:
                if type(key) == str:
                    key = "self.model_fit_params['{}']".format(key)

                warnings.simplefilter("always")
                warnings.warn("\n\nWarning: incompatible fit_params found for "
                              "model. Fit method will ignore parameters:"
                              "\n\n{0}\n\n"
                              "Please set {1} with compataible parameters to "
                              "use custom fit parameters for the model. If "
                              "{1} is not set, self.model_fit_params['final'] "
                              "will be tried by default."
                              .format(fit_params.keys(), key), stacklevel=9999)
                warnings.simplefilter("default")

                clf.fit(X, y)
        else:
            clf.fit(X, y)

        return clf

    def _fit_ovr(self, X, y, eval_set, ovr_vals, fbeta_weight, enc, attach,
                 model=None):
        '''
        Utility function for fitting or testing a model in a OVR fashion
        against a (possibly) classification-masked X-dataset.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Training input data.
        y: array-like, shape = [n_samples, ]
            True classification value for training data.
        eval_set: list of (X, y) pair
            Dataset to use as validation set for early-stopping and/or scoring
            trained models.
        ovr_vals: list
            For each value in ovr_vals, the y values is turned into True/False
            labels for OVR modeling. Subsequent OVR model training will mask
            out classifications used in previous ovr_val trainings.
        fbeta_weight: float
            The strength of recall versus precision in the F-score.
        enc: LabelEncoder
            Fitted LabelEncoder for storage when testing models.
        attach: boolean
            Whether to attach the fitted model to the OrderedOVRClassifier
            pipeline.
        model: model, optional
            A model to test against X dataset with masked classifications.

        Returns
        -------
        clf: OOVR_Model or None
            Returns the OVR fitted model when testing a model (input is not
            None). Otherwise, returns nothing.
        '''
        # Set mask to remove classes from future training steps.
        if self.mask is None:
            self.mask = np.zeros(len(y)).astype(bool)

        if eval_set is not None and self.eval_mask is None:
            self.eval_mask = np.zeros(len(eval_set[0][1])).astype(bool)

        for ovr_val in ovr_vals:
            Xm, ym, eval_X, eval_y = self._mask_datasets(X, y, eval_set,
                                                         ovr_val)
            clf, fit_params = self._get_model(ovr_val, eval_X, eval_y, model)
            clf = self._fit_model(clf, Xm, ym, ovr_val, fit_params)
            title = str(ovr_val).title()

            # Use best weighted fscore as threshold and set mask for future
            # steps to remove true values for class from subsequent training.
            if eval_set is None:
                best, scores = u.plot_thresholds(clf, Xm, ym, fbeta_weight,
                                                 '{} vs. Rest'.format(title))
            else:
                best, scores = u.plot_thresholds(clf, eval_X, eval_y,
                                                 fbeta_weight,
                                                 '{} vs. Rest'.format(title))

            model_attr = {
                'model': clf,
                'thresholds': best/100,
                'rest_precision': scores[0][0],
                'mask': np.logical_or(self.mask, y == ovr_val),
                '_le': enc
            }

            if eval_set is not None:
                eval_mask = np.logical_or(self.eval_mask,
                                          eval_set[0][1] == ovr_val)
                model_attr.update({'eval_mask': eval_mask})

            if attach:  # add fitted model for ovr_val to the pipeline
                self.__updateattr__(model_attr)

            else:  # model is being trained in test without pipeline attachment
                model_attr.update({'ovr_val': ovr_val})
                return u.OOVR_Model(model_attr)

            print
            print '-'*80
            print 'finished Ordered OVR fit for value: {}'.format(ovr_val)
            print '-'*80
            print

        return

    def _get_model(self, key, eval_X, eval_y, model=None):
        '''
        Retrieve model and fit_params from model_dict and model_fit_params
        attributes. Also adds eval_set to the fit_params for xgboost and
        lightgbm classification models.

        Parameters
        ----------
        key: str
            Label from dictionaries where model and fit_params will be fetched.
        eval_X: array-like, shape = [n_samples, n_features]
            X input evaluation data for early stopping.
        eval_y: array-like, shape = [n_samples, ]
            True classification values to score early stopping.
        model: model, optional
            A model to test against X dataset with masked classifications.
            If specified, does not retrieve model from model_dict with supplied
            key.

        Returns
        -------
        clf: model
            Unfitted model to use for later training.
        fit_params: dict
            Fit parameters to pass into clf.
        '''
        def check_eval_metric(clf, fit_params):
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
                fit_params['eval_set'] = [(eval_X, eval_y)]
                binary = len(np.unique(eval_y)) == 2

                if 'early_stopping_rounds' not in fit_params:
                    fit_params['early_stopping_rounds'] = 10

                if 'eval_metric' not in fit_params:
                    if binary:
                        fit_params['eval_metric'] = early_stop_berror[module]
                    else:
                        fit_params['eval_metric'] = early_stop_mlog[module]
                elif binary:
                    if fit_params['eval_metric'] in mlog:
                        fit_params['eval_metric'] = early_stop_berror[module]
                elif fit_params['eval_metric'] in berror:
                    fit_params['eval_metric'] = early_stop_mlog[module]

            return fit_params

        # If model_dict does not contain key passed into function, use final
        # model by default.
        default_m = self.model_dict['final']
        default_p = self.model_fit_params['final']

        if model is not None:  # Use test model specified by user.
            clf = model
        else:
            clf = copy.deepcopy(self.model_dict.get(key, default_m))

        fit_params = copy.deepcopy(self.model_fit_params.get(key, default_p))

        if eval_X is not None:
            fit_params = check_eval_metric(clf, fit_params)

        return clf, fit_params

    def _json_transform(self, row):
        '''
        Utility function for transforming JSON into a numpy array for single
        data point prediction.

        Parameters
        ----------
        row: json
            Single JSON row to make prediction from.

        Returns
        -------
        row: array-like, shape = [1, n_features]
            Numpy array to pass into predict function.
        '''
        if self.input_cols is None:
            raise RuntimeError("self.input_cols must be specified with the "
                               "correct column ordering for predictions from "
                               "JSON")

        row = pd.Series(json.loads(row), index=self.input_cols)

        if all(row.isna()):
            warnings.simplefilter('always')
            warnings.warn('\n\nWarning: Missing values found in input',
                          stacklevel=9999)
            warnings.simplefilter('default')

        row = row.values.reshape(1, -1)

        return row

    def _mask_datasets(self, X, y, eval_set, ovr_val=None):
        '''
        Filters out trained classes from the input dataset. If ovr_val is
        specified, sets y values to True/False for OVR. If no previous training
        was fitted and ovr_val is not specified, returns original data.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Training input data.
        y: array-like, shape = [n_samples, ]
            True classification value for training data.
        eval_set: list of (X, y) pair
            Dataset to use as validation set for early-stopping and/or scoring
            trained models.
        ovr_val: str, int, or float, optional
            Classification value to perform OVR training.

        Returns:
        --------
        Xm: array-like, shape = [<= n_samples, n_features]
            X with data from previously trained classes filtered out of data.
        ym: array-like, shape = [<= n_samples, ]
            y with data from previously trained classes filtered out of data.
        eval_X: array-like, shape = [<= n_samples, n_features]
            X from eval_set with data from previously trained classes filtered
            out of data.
        eval_y: array-like, shape = [<= n_samples, ]
            y from eval_set with data from previously trained classes filtered
            out of data.
        '''
        if self.mask is None:
            return X, y, eval_set[0][0], eval_set[0][1]

        Xm = X[~self.mask]
        ym = y[~self.mask]

        if ovr_val is not None:
            ym = ym == ovr_val

        if eval_set is not None:
            eval_X = eval_set[0][0][~self.eval_mask]
            eval_y = eval_set[0][1][~self.eval_mask]

            if ovr_val is not None:
                eval_y = eval_y == ovr_val
        else:
            eval_X, eval_y = None, None

        return Xm, ym, eval_X, eval_y

    def _pred_cleanup(self, X, drop_cols):
        '''
        Utility function for removing user specified drop_cols from a DataFrame
        input.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Data used for predictions.
        drop_cols: list of str, optional
            Labels of columns to ignore in modeling, only applicable to pandas
            DataFrame X input.

        Returns
        -------
        X: array-like, shape = [n_samples, <= n_features]
            Data with drop_cols removed.
        '''
        if self.pipeline[-1][0] != 'final':
            raise RuntimeError('A final model needs to be fit to make '
                               'predictions.')

        if X.__class__ == pd.DataFrame:
            if drop_cols is None:
                drop_cols = []

            if self.target in X.columns:
                drop_cols.append(self.target)

            X = X.drop(drop_cols, axis=1)

            if self.input_cols is not None:
                if not all(self.input_cols == X.columns):
                    raise AssertionError("Incompatible columns! Please check "
                                         "if X columns match self.input_cols.")

        return X

    def _xg_cleanup(self, clf):
        '''
        Utility function to delete the Booster.feature_names attributes in
        XGBClassifier. Deleting this attribute allows XGBClassifier to make
        predictions from either a numpy array or DataFrame input.

        Parameters
        ----------
        clf: model
            Model used to make predictions.
        '''
        if clf.__module__ == 'xgboost.sklearn':
            if clf._Booster.feature_names:
                del clf._Booster.feature_names

    def _xy_transform(self, X, y, drop_cols=None):
        '''
        Utility function for removing user specified drop_cols from a DataFrame
        input. Also extracts y from a DataFrame when an empty y input is
        provided.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Input data. Can be DataFrame (with or without target column
            included) or numpy array.
        y: array-like, shape = [n_samples, ] or None
            True labels for X. If not provided and X is a DataFrame, column
            specified by self.target is extracted as the y output.
        drop_cols: list of str, optional
            Labels of columns to ignore in modeling, only applicable to pandas
            DataFrame X input.

        Returns
        -------
        X: array-like, shape = [n_samples, <= n_features]
            Data for future modeling/predicting steps, with (if applicable)
            target and drop columns removed.
        y: array-like, shape = [n_samples, ]
            True labels for X.
        drop_cols: list of str
            Labels of columns to ignore in modeling, plus label of target
            variable if specifed on init, only applicable to pandas DataFrame X
            input.
        '''
        if drop_cols is None:
            drop_cols = []

        if self.target:
            drop_cols.append(self.target)

        if y is not None:
            y = pd.Series(y)
        elif self.target:  # set y from pandas DataFrame if self.target is set
            y = X[self.target].copy()
        else:
            raise AssertionError('Please initiate class with a label for '
                                 'target variable OR input a y parameter '
                                 'into fit')

        assert len(X) == len(y)

        # Drop columns if X is DataFrame and drop_cols is specified
        if X.__class__ == pd.DataFrame:
            X = X.drop(drop_cols, axis=1)

            if self.input_cols is None:
                self.input_cols = X.columns
            else:
                if not all(self.input_cols == X.columns):
                    raise AssertionError("Incompatible columns! Please check "
                                         "if X columns match self.input_cols.")

        return X, y, drop_cols

    def attach_model(self, oovr_model):
        '''
        Attaches an OVR model to the OrderedOVRClassifier prediction pipeline

        Parameters
        ----------
        oovr_model: OOVR_Model
            OOVR_Model object returned from fit_test of fit_test_ovr functions.
            OOVR_Model contains compatible OVR classifer to add to the
            prediction pipeline of OrderedOVRClassifier.

        Returns
        -------
        self
        '''
        if len(self.pipeline) > 0:
            if self.pipeline[-1][0] == 'final':
                raise RuntimeError('Attach failed. Final model already '
                                   'trained.')

        if hasattr(oovr_model, 'ovr_val'):
            ovr_val = oovr_model.ovr_val
            if ovr_val in self.ovr_vals:
                raise RuntimeError('Attach failed. Model already trained for '
                                   '{}'.format(ovr_val))

        self.__updateattr__(oovr_model.__dict__)

        return self

    def fit(self, X, y=None, eval_set=None, drop_cols=None):
        '''
        Fits OrderedOVRClassifier and attaches trained models to the class
        pipeline. If self.train_final_only == True (not default), fit skips
        the Ordered OVR training and trains/evaluates the model using the API
        for OrderedOVRClassifier on all classes. If self.train_final_model ==
        True (default), fit does training on remaining classes not specified in
        self.ovr_vals.

        Binary models are evaluated with the imported plot_thresholds function,
        which evaluates precision, recall, and fscores for all thresholds
        with 0.01 interval spacing and automatically sets the threshold at the
        best weighted fscore. Multiclass models are evaluated using the
        imported extended_classification_report function.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Input data for model training.
        y: array-like, shape = [n_samples, ], optional
            True labels for X. If not provided and X is a DataFrame, will
            extract y column from X with the provided self.target value.
        eval_set: DataFrame or list of (X, y) pair, optional
            Dataset to use as validation set for early-stopping and/or scoring
            trained models.
        drop_cols: list of str, optional
            Labels of columns to ignore in modeling, only applicable to pandas
            DataFrame X input.

        Returns
        -------
        self
        '''
        start = datetime.datetime.now()

        X, y, drop_cols = self._xy_transform(X, y, drop_cols)
        eval_set = self._eval_set(eval_set, drop_cols)
        enc = self._le.fit(y)  # fit LabelEncoder

        if not self.train_final_only:
            if len(self.ovr_vals) == 0:
                # If not specified, sets ovr_vals as the majority class in the
                # target variable by default. In practice, this just ensures
                # that a positive prediction for the majority class will always
                # win in the final multi-class classification.
                self.ovr_vals = [y.value_counts().index.tolist()[0]]

            # run _fit_ovr
            ovr_vals = self.ovr_vals
            fbeta_weight = self.fbeta_weight
            self._fit_ovr(X, y, eval_set, ovr_vals, fbeta_weight, enc,
                          attach=True)

        if self.train_final_model or self.train_final_only:
            # Fit the final model and attach to pipeline
            self._fit_final_model(X, y, eval_set, attach=True, model=None)

            if eval_set is not None:
                y_true = eval_set[0][1]
                y_pred = self.predict(eval_set[0][0])
            else:
                y_true = y
                y_pred = self.predict(X)

            print
            print '-'*80
            print 'OVERALL REPORT'
            print
            u.extended_classification_report(y_true, y_pred)
            stop = datetime.datetime.now()
            duration = stop-start
            print('total training time: {0:.1f} minutes'
                  .format(duration.total_seconds()/60))
            print
            print '-'*80

        return self

    def fit_test(self, model, X, y=None, eval_set=None, drop_cols=None):
        '''
        Function for training a final model against a (possibly) classification
        masked X dataset. Does not attach trained model to the pipeline for
        OrderedOVRClassifier. Also evaluates classification with the imported
        extended_classification_report function.

        Parameters
        ----------
        model: model
            Unfitted model to test against dataset, which may have
            classification values masked if previous OVR training has been
            attached to pipeline.
        X: array-like, shape = [n_samples, n_features]
            Input data for model training.
        y: array-like, shape = [n_samples, ], optional
            True labels for X. If not provided and X is a DataFrame, will
            extract y column from X with the provided self.target value.
        eval_set: DataFrame or list of (X, y) pair, optional
            Dataset to use as validation set for early-stopping and/or scoring
            trained models.
        drop_cols: list of str, optional
            Labels of columns to ignore in modeling, only applicable to pandas
            DataFrame X input.

        Returns
        -------
        model: OOVR_Model
            OVR fitted model trained against classification-masked X dataset.
        '''
        X, y, drop_cols = self._xy_transform(X, y, drop_cols)
        eval_set = self._eval_set(eval_set, drop_cols)

        if len(self.pipeline) > 0:
            enc = LabelEncoder().fit(y)

        model = self._fit_final_model(X, y, eval_set, attach=False,
                                      model=model)

        return model

    def fit_test_grid(self, grid_model, X, y=None, eval_set=None,
                      ovr_val=None, drop_cols=None):
        '''
        Wrapper for testing hyper-parameter optimization models with the
        OrderedOVRClassifier API against a (possibly) classification-masked X
        dataset.

        Parameters
        ----------
        grid_model: GridSearchCV or RandomizedSearchCV model
            Hyper-parameter optimizer model from the sklearn.model_selection
            library. Must be initiated with base estimator and parameter grid.
        X: array-like, shape = [n_samples, n_features]
            Input data for model training.
        y: array-like, shape = [n_samples, ], optional
            True labels for X. If not provided and X is a DataFrame, will
            extract y column from X with the provided self.target value.
        eval_set: DataFrame or list of (X, y) pair, optional
            Dataset to use as validation set for early-stopping and/or scoring
            trained models.
        ovr_val: str, int, or float, optional
            If specified, fit_test_grid will perform OVR modeling against the
            ovr_val classification label.
        drop_cols: list of str, optional
            Labels of columns to ignore in modeling, only applicable to pandas
            DataFrame X input.

        Returns
        -------
        grid_model: GridSearchCV or RandomizedSearchCV model
            Hyper-parameter optimizer model with recorded optimization results.
            Note that by design, retrain is set to False, and the user will
            need to train a new model with the best parameters found if they
            choose to attach the model to the OrderedOVRClassifier pipeline.
        '''
        if grid_model.estimator.__module__ == 'lightgbm.sklearn':
            lgbm_grid = True
        else:
            lgbm_grid = False

        X, y, drop_cols = self._xy_transform(X, y, drop_cols)
        eval_set = self._eval_set(eval_set, drop_cols)
        Xm, ym, eval_X, eval_y = self._mask_datasets(X, y, eval_set, ovr_val)

        if not ovr_val:
            ym, eval_y, _ = self._encode_y(ym, eval_y, lgbm_grid)

        _, fit_params = self._get_model('final', eval_X, eval_y, grid_model)

        if eval_X is not None:
            if X.__class__ == pd.DataFrame:
                X = Xm.append(eval_X)
            elif X.__class__ == np.ndarray:
                X = np.vstack([Xm, eval_X])

            y = np.hstack([ym, eval_y])
            cv = [(np.arange(len(ym)), np.arange(len(ym), len(y)))]
            grid_model.set_params(cv=cv, refit=False)

        grid_model = self._fit_model(grid_model, X, y, 'final', fit_params)

        print
        print '-'*80
        print 'best_params:\n{}\n'.format(grid_model.best_params_)
        print 'best {0} score:\n{1}'.format(grid_model.scoring,
                                            grid_model.best_score_)
        print '-'*80
        print

        return grid_model

    def fit_test_ovr(self, model, ovr_val, X, y=None, eval_set=None,
                     drop_cols=None):
        '''
        Function for training an OVR model against a (possibly) classification
        masked X dataset. Does not attach trained model to the pipeline for
        OrderedOVRClassifier. Also evaluates binary classification with the
        imported plot_thresholds function, which plots precision, recall, and
        fscores for all thresholds with 0.01 interval spacing.

        Parameters
        ----------
        model: model
            Unfitted model to test against dataset, which may have
            classification values masked if previous OVR training has been
            attached to pipeline.
        ovr_val: str, int, or float
            Classification value to perform OVR training.
        X: array-like, shape = [n_samples, n_features]
            Input data for model training.
        y: array-like, shape = [n_samples, ], optional
            True labels for X. If not provided and X is a DataFrame, will
            extract y column from X with the provided self.target value.
        eval_set: DataFrame or list of (X, y) pair, optional
            Dataset to use as validation set for early-stopping and/or scoring
            trained models.
        drop_cols: list of str, optional
            Labels of columns to ignore in modeling, only applicable to pandas
            DataFrame X input.

        Returns
        -------
        model: OOVR_Model
            OVR fitted model trained against classification-masked X dataset.
        '''
        X, y, drop_cols = self._xy_transform(X, y, drop_cols)
        eval_set = self._eval_set(eval_set, drop_cols)
        fbeta_weight = self.fbeta_weight

        assert ovr_val in y.unique()

        # fit LabelEncoder if not fitted yet, otherwise take existing
        if hasattr(self._le, 'classes_'):
            enc = self._le
        else:
            enc = self._le.fit(y)

        model = self._fit_ovr(X, y, eval_set, [ovr_val], fbeta_weight, enc,
                              attach=False, model=model)

        return model

    def plot_oovr_dependencies(self, ovr_val, X, y=None, comp_vals=None,
                               drop_cols=None):
        '''
        Evaluates the effect of changing the threshold of an ordered OVR
        classifier against other classes with respect to accuracy, precision,
        recall, and f1 metrics.

        Parameters
        ----------
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
        drop_cols: list of str, optional
            Labels of columns to ignore in modeling, only applicable to pandas
            DataFrame X input.
        '''
        X, y, _ = self._xy_transform(X, y, drop_cols)
        return u.plot_oovr_dependencies(self, ovr_val, X, y, comp_vals)

    def predict(self, X, start=0, drop_cols=None):
        '''
        Predict multi-class targets using underlying estimators. Positive
        predictions from earlier steps in the prediction pipeline will be the
        final prediction, as this is the intended functionality of
        OrderedOVRClassifier.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Data used for predictions.
        start: int, optional, default: 0
            Index of the prediction pipeline to start on. Defaults to 0 (makes
            prediction through full pipeline).
        drop_cols: list of str, optional
            Labels of columns to ignore in modeling, only applicable to pandas
            DataFrame X input.

        Returns
        -------
        pred: array-like, shape = [n_samples, ]
            Predicted multi-class targets.
        '''
        X = self._pred_cleanup(X, drop_cols)

        # Get predictions for final model
        clf = self.pipeline[-1][1]
        self._xg_cleanup(clf)
        pred = self.pipeline[-1][1].predict(X)

        # Inverse label transform
        pred = self._le.inverse_transform(pred)

        # Set values to ordered OVR predictions
        for i in np.arange(len(self.ovr_vals))[start:][::-1]:
            clf = self.pipeline[i][1]
            self._xg_cleanup(clf)

            thld = self.thresholds[i]
            mask = clf.predict_proba(X)[:, 1] >= thld
            pred[mask] = self.ovr_vals[i]

        return pred

    def predict_proba(self, X, drop_cols=None):
        '''
        Predict probabilities for multi-class targets using underlying
        estimators. If positive predictions are found in earlier steps of the
        pipeline, the probabilities are normalized so that the probability of
        positive prediction is reflected in the final probabilites. If negative
        predictions are found in earlier steps of the pipeline, this function
        returns a dumb guesstimate of the class probability: the minimum of
        1/n_features and 1-precision@threshold for the rest classification.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Data used for predictions.
        drop_cols: list of str, optional
            Labels of columns to ignore in modeling, only applicable to pandas
            DataFrame X input.

        Returns
        -------
        pred: array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in self._le.classes_.
        '''
        X = self._pred_cleanup(X, drop_cols)
        n, m = len(X), len(self._le.classes_)
        ovr_cols = self._le.transform(self.ovr_vals)

        # Get predictions for final model
        clf = self.pipeline[-1][1]
        self._xg_cleanup(clf)
        probas = clf.predict_proba(X)
        current_features = probas.shape[1]

        # Initiate new array
        pred = np.zeros((n, m))
        proba_cols = np.setdiff1d(np.arange(m), ovr_cols)
        pred[:, proba_cols] = probas

        # Set probabilities to ordered OVR predictions
        for i in np.arange(len(self.ovr_vals))[::-1]:
            current_features += 1

            clf = self.pipeline[i][1]
            self._xg_cleanup(clf)
            ovr_proba = clf.predict_proba(X)[:, -1]

            thld = self.thresholds[i]
            guess_prob = np.min([1 - self.rest_precision[i],
                                 1 / current_features])

            # Recalibrate probability so that positive class predict_proba for
            # OVR falls within the [0.5, 1.0] range for thresholds below 0.5
            if thld < 0.5:
                ovr_proba = 0.5 + (ovr_proba - thld) / (2 * (1-thld))
                mask = ovr_proba >= 0.5
            else:
                mask = ovr_proba >= thld

            # Division to ensure probabilites are correct after normalization
            ovr_proba = ovr_proba / (1 - ovr_proba + 1e-10)

            # Set probabilities and normalize. guess_prob for negative
            # predictions will always be equal or less than the normalized
            # average
            pred[mask, ovr_cols[i]] = ovr_proba[mask]
            pred[~mask, :] = pred[~mask, :] * (1-guess_prob)
            pred[~mask, ovr_cols[i]] = guess_prob
            pred /= np.sum(pred, axis=1)[:, np.newaxis]

        return pred

    def predict_json(self, row):
        '''
        Predict multi-class target from JSON using underlying estimators.
        Positive predictions from earlier steps in the prediction pipeline
        will be the final prediction, as this is the intended functionality of
        OrderedOVRClassifier.

        Parameters
        ----------
        row: json
            Single JSON row to make prediction from.

        Returns
        -------
        pred: str or int
            Predicted multi-class target for input row data.
        '''
        row = self._json_transform(row)

        for i in xrange(len(self.ovr_vals)):
            clf = self.pipeline[i][1]
            self._xg_cleanup(clf)
            pred = clf.predict(row)

            if pred[0] == 1:
                return self.ovr_vals[i]

        clf = self.pipeline[-1][1]
        self._xg_cleanup(clf)
        pred = clf.predict(row)
        pred = self._le.inverse_transform(pred)[0]

        return pred

    def predict_proba_json(self, row, print_prob=False):
        '''
        Predict probabilities for multi-class target from JSON using
        underlying estimators. If positive predictions are found in earlier
        steps of the pipeline, the probabilities are normalized so that the
        probability of positive prediction is reflected in the final
        probabilites. If negative predictions are found in earlier steps of the
        pipeline, this function returns a dumb guesstimate of the class
        probability: the minimum of 1/n_features and 1-precision@threshold for
        the rest classification.

        Parameters
        ----------
        row: json
            Single JSON row to make prediction from.

        Returns
        -------
        pred: array-like, shape = [1, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in self._le.classes_.
        '''
        row = self._json_transform(row)
        pred = self.predict_proba(row)

        if print_prob:
            ljust = max([len(x) for x in self._le.classes_]) + 1
            for i, p in enumerate(pred[0]):
                print "{}: {:.3f}".format(self._le.classes_[i].ljust(ljust), p)

        return pred

    def score(self, X, y=None, sample_weight=None, drop_cols=None):
        '''
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Test samples.
        y: array-like, shape = [n_samples, ], optional
            True labels for X.
        sample_weight: array-like, shape = [n_samples], optional
            Sample weights.
        drop_cols: list of str, optional
            Labels of columns to ignore in modeling, only applicable to pandas
            DataFrame X input.

        Returns
        -------
        scr: float
            Mean accuracy of self.predict(X) wrt y.
        '''
        X, y, _ = self._xy_transform(X, y, drop_cols)
        scr = m.accuracy_score(y, self.predict(X), sample_weight=sample_weight)

        return scr
