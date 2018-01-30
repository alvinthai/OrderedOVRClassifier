"""
OrderedOVRClassifier
"""

# Author: Alvin Thai <alvinthai@gmail.com>

from __future__ import division, print_function
from itertools import repeat
import copy
import datetime
import json
import logging
import numpy as np
import oovr_utils as u
import pandas as pd
import sklearn.metrics as m

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

    1. Reduce training time by screening out classes that can be predicted with
       high confidence. Most multi-classification predictive models use
       One-Vs-Rest approaches and the predicted outcome is simply the highest
       predicted (normalized) probability among all One-Vs-Rest trained models.
       If a highly predictive class exists among the multi-classification
       labels, it could be advantageous to remove the highly predictive class
       from future steps to reduce training time (by virtue of having less
       classes to make predictions on) for later modeling steps that may
       require heavy optimization.

    2. Evaluating the thresholds of a binary classifier against classification
       metrics of all downstream predictive models. With Ordered One-Vs-Rest
       Classification, the binary outcome from an Ordered One-Vs-Rest model can
       be optimized to achieve an ideal mix of accuracy/precision/recall scores
       among each predictive class. Call the plot_oovr_dependencies function on
       a fully trained OrderedOVRClassifier model to execute these evaluations.

    3. Mixing different classification algorithms. With an Ordered One-Vs-Rest
       approach, the same machine learning algorithm is not required to be used
       for predicting all classes. This gives the user more flexibility to
       explore different combinations of algorithms to make final predictions.

    The API for OrderedOVRClassifier is designed to be user-friendly with
    pandas, numpy, and scikit-learn. There is built in functionality to support
    easy handling for early stopping on the sklearn wrapper for XGBoost and
    LightGBM. If working with DataFrames, fitting a model with early stopping
    could be done using a command as simple as:

    > OrderedOVRClassifier(target='label').fit(X=train_df, eval_set=eval_df)

    OrderedOVRClassifier runs custom evaluation functions to diagnose and/or
    plot the predictive performance of the classification after training each
    model. Additionally, a grid search wrapper is built into the API for
    hyper-parameter tuning against classification-subsetted datasets.

    OrderedOVRClassifier also includes utilities for model agnostic evaluation
    of feature importances and partial dependence. These model agnostic
    evaluation utilities (plot_feature_importance and plot_partial_dependence)
    require the skater library and are approximations based on a random sample
    of the data.

    Parameters
    ----------
    target: str
        Label for target variable in pandas DataFrame. If provided, all future
        future inputs with an X DataFrame do not require an accompanying y
        input, as y will be extracted from the X DataFrame; however, the target
        column must be included in the X DataFrame for all fitting steps if the
        target parameter is provided.

    ovr_vals: list
        List of target values (and ordering) to perform ordered one-vs-rest.

    model_dict: dict of models
        Dictionary of models to perform ordered one-vs-rest, dict should
        include a model for each value in ovr_vals, and a model specified for
        the last classifier named 'final'.

        i.e. model_dict = { value1: LogisticRegression(),
                            value2: RandomForestClassifier(),
                           'final': XGBClassifier() }

    model_fit_params: dict of dict
        Additional parameters (inputted as a dict) to pass to the fit step of
        the models specified in self.model_dict.

        i.e. model_fit_params = {'final': {'verbose': False} }

    Attributes
    ----------
    input_cols: list of str, shape = [n_features]
        List of columns saved when initially fitting OrderedOVRClassifier.
        Column headers for future X and eval_set inputs (if trained using
        DataFrame) must match self.input_cols for training or prediction to
        proceed.

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
                 model_fit_params=None):
        self.target = target
        self.ovr_vals = ovr_vals
        self.model_dict = model_dict
        self.model_fit_params = model_fit_params

        self._default_attributes()
        self._init_setup_defaults()

        self._logger = logging.Logger('')
        self._logger.addHandler(logging.StreamHandler())

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
            if ovr_val not in self.ovr_vals:
                self.__getattribute__('ovr_vals').append(ovr_val)

    def _check_ovr(self, y):
        '''
        Checks if self.ovr_vals contains valid classes. If self.ovr_vals
        indicates the whole pipeline is a series of binary classifications,
        the self._all_binary flag is set to True to indicate the final model
        is not multiclass.

        Parameters
        ----------
        y: array-like, shape = [n_samples, ]
            True classification values for training data.

        Returns
        ------
        ovr_vals: list
            List of target values (and ordering) to perform ordered one-vs-rest
        '''
        classes = set(y.unique())
        n_classes = len(classes)
        n_foreign = len(set(self.ovr_vals) - classes)
        n_remaining = len(classes - set(self.ovr_vals))

        if n_foreign > 0:
            msg = 'Invalid values passed in ovr_vals during initialization.'
            raise RuntimeError(msg)

        if n_remaining <= 1:
            self.ovr_vals = self.ovr_vals[:n_classes-1]
            self._all_binary = True

        return self.ovr_vals

    def _default_attributes(self):
        '''
        Sets/resets defualt attributes for OrderedOVRClassifier.
        '''
        self._le = LabelEncoder()
        self._all_binary = False
        self._eval_mask = None
        self._mask = None
        self.input_cols = None
        self.pipeline = []
        self.rest_precision = []
        self.thresholds = []

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

        lgmb_grid: boolean, optional, default: False
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
        eval_set is a DataFrame, unpacks it into a list containing an (X, y)
        tuple. Aside from xgboost/lightgbm, eval_set is also used to evaluate
        trained models on unseen data and validate grid searches.

        Parameters
        ----------
        eval_set: DataFrame or list of (X, y) tuple
            Dataset to use as validation set for early-stopping and/or scoring
            trained models.

        drop_cols: list of str
            Columns to drop from DataFrame prior to modeling.

        Returns
        -------
        eval_set: list of (X, y) tuple
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

    def _fit_final_model(self, X, y, eval_set, attach, model=None,
                         train_final_only=False, fit_params=None):
        '''
        Utility function for fitting final model or testing a model against a
        classification-masked X dataset.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Training input data.

        y: array-like, shape = [n_samples, ]
            True classification values for training data.

        eval_set: list of (X, y) tuple
            Dataset to use as validation set for early-stopping and/or scoring
            trained models.

        attach: boolean
            Whether to attach the fitted model to the OrderedOVRClassifier
            pipeline.

        model: model, optional
            A model to test against X dataset with masked classifications.

        train_final_only: bool, optional, default: False
            Whether OVR modeling was ignored.

        fit_params: dict, optional
            Key-value pairs of optional arguments to pass into model fit
            function.

        Returns
        -------
        clf: model
            Fitted model trained against classification-masked X dataset.
        '''
        Xm, ym, eval_X, eval_y = self._mask_datasets(X, y, eval_set)
        ym, eval_y, enc = self._encode_y(ym, eval_y)
        clf, fit_params = self._get_model('final', eval_X, eval_y, model,
                                          fit_params)

        clf = self._fit_model(clf, Xm, ym, 'final', fit_params)

        if eval_set is not None:
            y_true = eval_y
            y_pred = clf.predict(eval_X)
        else:
            y_true = ym
            y_pred = clf.predict(Xm)

        y_true = enc.inverse_transform(y_true)
        y_pred = enc.inverse_transform(y_pred)

        if not train_final_only:
            print('', '-'*80, sep='\n')
            u.extended_classification_report(y_true, y_pred)

        if attach:
            self.pipeline.append(('final', clf))

            if not train_final_only:
                prt = 'finished fit for remaining classes'
                print('-'*80, prt, '-'*80, '', sep='\n')
        else:
            fmodel_attr = {'model': clf, '_le': enc}
            return u.OOVR_Model(fmodel_attr)

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
            True classification values for training data.

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

                msg = str("Warning: incompatible fit_params found for model. "
                          "Fit method will ignore parameters:\n\n{0}\n\n"
                          "Please set {1} with compataible parameters to use "
                          "custom fit parameters for the model. If {1} is not "
                          "set, self.model_fit_params['final'] will be tried "
                          "by default.".format(fit_params.keys(), key))

                self._logger.warn(msg)

                clf.fit(X, y)
        else:
            clf.fit(X, y)

        return clf

    def _fit_ovr(self, X, y, eval_set, ovr_vals, fbeta_weight, enc, attach,
                 model=None, fit_params=None):
        '''
        Utility function for fitting or testing a model in a OVR fashion
        against a (possibly) classification-masked X-dataset.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Training input data.

        y: array-like, shape = [n_samples, ]
            True classification values for training data.

        eval_set: list of (X, y) tuple
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

        fit_params: dict, optional
            Key-value pairs of optional arguments to pass into model fit
            function.

        Returns
        -------
        clf: OOVR_Model or None
            Returns the OVR fitted model when testing a model (input is not
            None). Otherwise, returns nothing.
        '''
        # Set mask to remove classes from future training steps.
        if self._mask is None:
            self._mask = np.zeros(len(y)).astype(bool)

        if eval_set is not None and self._eval_mask is None:
            self._eval_mask = np.zeros(len(eval_set[0][1])).astype(bool)

        for ovr_val in ovr_vals:
            if attach and fit_params is not None:
                # attach == True when fit (and not fit_test_ovr) is called
                # if specified, fit_params is expected to be dict of dict
                fit_ovr_params = fit_params.get(ovr_val, None)
            else:
                fit_ovr_params = fit_params

            Xm, ym, eval_X, eval_y = self._mask_datasets(X, y, eval_set,
                                                         ovr_val)
            clf, fit_ovr_params = self._get_model(ovr_val, eval_X, eval_y,
                                                  model, fit_ovr_params)
            clf = self._fit_model(clf, Xm, ym, ovr_val, fit_ovr_params)
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
                'ovr_val': ovr_val,
                'rest_precision': scores[0][0],
                'thresholds': best/100,
                '_le': enc,
                '_mask': np.logical_or(self._mask, y == ovr_val)
            }

            if eval_set is not None:
                eval_mask = np.logical_or(self._eval_mask,
                                          eval_set[0][1] == ovr_val)
                model_attr.update({'_eval_mask': eval_mask})

            if attach:  # add fitted model for ovr_val to the pipeline
                self.__updateattr__(model_attr)

            else:  # model is being trained in test without pipeline attachment
                return u.OOVR_Model(model_attr)

            prt = 'finished Ordered OVR fit for value: {}'.format(ovr_val)
            print('', '-'*80, prt, '-'*80, '', sep='\n')

        if self._all_binary:
            val = list(set(enc.classes_) - set(self.ovr_vals))
            val = self._le.transform(val)[0]
            fmodel_attr = {'model': u.UniformClassifier(val)}
            self.__updateattr__(fmodel_attr)

        return

    def _get_model(self, key, eval_X, eval_y, model=None, fit_params=None,
                   prefix=None):
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
            A model to test against X dataset with masked classifications. If
            specified, does not retrieve model from model_dict with supplied
            key.

        fit_params: dict, optional
            Key-value pairs of optional arguments to pass into model fit
            function.

        prefix: str, optional
            String prefix for parameter keys.

        Returns
        -------
        clf: model
            Unfitted model to use for later training.

        fit_params: dict
            Fit parameters to pass into clf.
        '''
        # If model_dict does not contain key passed into function, use final
        # model by default.
        default_m = self.model_dict['final']
        default_p = self.model_fit_params['final']

        if model is not None:  # Use test model specified by user.
            clf = model
        else:
            clf = copy.deepcopy(self.model_dict.get(key, default_m))

        base_params = copy.deepcopy(self.model_fit_params.get(key, default_p))

        # prefix keys when model is a grid serach with pipeline estimator
        if type(prefix) == str:
            for key in base_params:
                base_params[prefix + key] = base_params.pop(key)

        if type(fit_params) == dict:
            base_params.update(fit_params)
        elif fit_params is not None:
            msg = 'fit_params not a dictionary. ignoring fit_params...'
            self._logger.warn(msg)

        if eval_X is not None:
            fit_params = u.check_eval_metric(clf, base_params, eval_X, eval_y,
                                             prefix=prefix)
        else:
            fit_params = base_params

        return clf, fit_params

    def _init_setup_defaults(self):
        '''
        Function for assigning default values when user does not supply values
        into the __init__ of OrderedOVRClassifier.
        '''
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
            self._logger.warn('Warning: Missing values found in input')

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
            True classification values for training data.

        eval_set: list of (X, y) tuple
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
        if self._mask is None:
            if eval_set is not None:
                return X, y, eval_set[0][0], eval_set[0][1]
            else:
                return X, y, None, None

        Xm = X[~self._mask]
        ym = y[~self._mask]

        if ovr_val is not None:
            ym = ym == ovr_val

        if eval_set is not None:
            eval_X = eval_set[0][0][~self._eval_mask]
            eval_y = eval_set[0][1][~self._eval_mask]

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
            Labels of columns ignored in modeling, only applicable to pandas
            DataFrame X input.

        Returns
        -------
        X: array-like, shape = [n_samples, <= n_features]
            Data with drop_cols removed.
        '''
        if self.pipeline[-1][0] != 'final':
            msg = 'A final model needs to be fit to make predictions.'
            raise RuntimeError(msg)

        pipeline_ovrs = [x[0] for x in self.pipeline[:-1]]

        if pipeline_ovrs != self.ovr_vals:
            raise RuntimeError('Bad Pipeline! Please fit/attach OrderedOVR'
                               'Classifier in the sequence specified '
                               'by the user-inputted ovr_vals.')

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

    def _skater_extract(self, X, n_jobs):
        '''
        Utility function passing parameters over to skater Interpretation and
        InMemoryModel classes.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Input data used for training or evaluating the fitted model.

        n_jobs: int, optional, default: -1
            The number of CPUs to use to compute with Skater. -1 means
            'all CPUs' (default).

        Returns
        -------
        X: array-like, shape = [n_samples, n_features]
            Input data used for training or evaluating the fitted model,
            converted to a numpy array.

        n_jobs: int
            The number of CPUs to use to compute with Skater. -1 means
            'all CPUs' (default). Changes to 1 for models that don't support
            multipooling.

        col_names: list
            Names to call features.
        '''
        if X.__class__ == pd.DataFrame:
            col_names = X.columns
            X = X.values
        else:
            col_names = None

        no_multipooling = ['lightgbm.sklearn', 'xgboost.sklearn',
                           'sklearn.linear_model.logistic']

        for _, clf in self.pipeline:
            if clf.__module__ in no_multipooling:
                n_jobs = 1
                break

        return X, n_jobs, col_names

    @staticmethod
    def _xg_cleanup(clf):
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

        if self.target is not None:
            drop_cols.append(self.target)

        if y is not None:
            y = pd.Series(y)
        elif self.target is not None:
            # set y from pandas DataFrame if self.target is set
            y = X[self.target].copy()
        else:
            raise RuntimeError('Please initiate class with a label for target '
                               'variable OR input a y parameter into fit')

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
        Attaches an OVR model to the OrderedOVRClassifier prediction pipeline.

        Parameters
        ----------
        oovr_model: OOVR_Model
            OOVR_Model object returned from fit_test of fit_test_ovr functions.
            OOVR_Model contains compatible OVR classifier to add to the
            prediction pipeline of OrderedOVRClassifier.

        Returns
        -------
        self
        '''
        if len(self.pipeline) > 0:
            if self.pipeline[-1][0] == 'final':
                msg = 'Attach failed. Final model already trained'
                raise RuntimeError(msg)

        if hasattr(oovr_model, 'ovr_val'):
            ovr_val = oovr_model.ovr_val
            if ovr_val in self.ovr_vals:
                msg = 'Attach failed. Model already trained for {}'
                msg = msg.format(ovr_val)
                raise RuntimeError(msg)

        self.__updateattr__(oovr_model.__dict__)

        return self

    def fit(self, X, y=None, eval_set=None, drop_cols=None, fbeta_weight=1.0,
            train_final_model=True, train_final_only=False,
            model_fit_params=None):
        '''
        Fits OrderedOVRClassifier and attaches trained models to the class
        pipeline.

        If train_final_only=True (not default), fit skips the Ordered OVR
        training and trains/evaluates the model using the API for
        OrderedOVRClassifier on all classes.

        If train_final_model=True (default), fit does training on remaining
        classes not specified in self.ovr_vals.

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

        eval_set: DataFrame or list of (X, y) tuple, optional
            Dataset to use as validation set for early-stopping and/or scoring
            trained models.

        drop_cols: list of str, optional
            Labels of columns to ignore in modeling, only applicable to pandas
            DataFrame X input.

        fbeta_weight: float, optional, default: 1.0
            The strength of recall versus precision in the F-score.

        train_final_model: bool, optional, default: True
            Whether to train a final model to the remaining data after OVR fits

        train_final_only: bool, optional, default: False
            Whether to ignore OVR modeling and to train the final model only.

        model_fit_params: dict of dict, optional
            Additional parameters (inputted as a dict) to pass to the fit step
            of the models specified in self.model_dict.

            i.e. model_fit_params = {'final': {'verbose': False} }

        Returns
        -------
        self
        '''
        start = datetime.datetime.now()
        self._default_attributes()

        X, y, drop_cols = self._xy_transform(X, y, drop_cols)
        eval_set = self._eval_set(eval_set, drop_cols)
        enc = self._le.fit(y)  # fit LabelEncoder

        if not train_final_only:
            if len(self.ovr_vals) == 0:
                # If not specified, sets ovr_vals as the majority class in the
                # target variable by default. In practice, this just ensures
                # that a positive prediction for the majority class will always
                # win in the final multi-class classification.
                self.ovr_vals = [y.value_counts().index.tolist()[0]]

            # run _fit_ovr
            ovr_vals = self._check_ovr(y)
            self._fit_ovr(X, y, eval_set, ovr_vals, fbeta_weight, enc,
                          attach=True, fit_params=model_fit_params)
        else:
            self.ovr_vals = []

        if train_final_model or train_final_only:
            if not self._all_binary:
                # if specified, model_fit_params is expected to be dict of dict
                if model_fit_params is not None:
                    fit_params = model_fit_params.get('final', None)
                else:
                    fit_params = None

                # Fit the final model and attach to pipeline
                self._fit_final_model(X, y, eval_set, attach=True,
                                      train_final_only=train_final_only,
                                      fit_params=fit_params)

            if eval_set is not None:
                y_true = eval_set[0][1]
                y_pred = self.predict(eval_set[0][0])
            else:
                y_true = y
                y_pred = self.predict(X)

            print('-'*80, 'OVERALL REPORT', '', sep='\n')

            u.extended_classification_report(y_true, y_pred)
            stop = datetime.datetime.now()
            duration = (stop-start).total_seconds()/60

            prt = 'total training time: {0:.1f} minutes'.format(duration)
            print(prt, '', '-'*80, sep='\n')

        return self

    def fit_test(self, model, X, y=None, eval_set=None, drop_cols=None,
                 fit_params=None):
        '''
        Function for training a final model against a (possibly) classification
        masked X dataset. Does not attach trained model to the pipeline for
        OrderedOVRClassifier. Also evaluates classification with the imported
        extended_classification_report function.

        Note that if an OVR model has been attached to the pipeline, the same
        dataset(s) used to train/evaluate the first OVR model must be used to
        train future OrderedOVRClassifier pipeline steps.

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

        eval_set: DataFrame or list of (X, y) tuple, optional
            Dataset to use as validation set for early-stopping and/or scoring
            trained models.

        drop_cols: list of str, optional
            Labels of columns to ignore in modeling, only applicable to pandas
            DataFrame X input.

        fit_params: dict, optional
            Key-value pairs of optional arguments to pass into model fit
            function.

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
                                      model=model, fit_params=fit_params)

        return model

    def fit_test_grid(self, grid_model, X, y=None, eval_set=None,
                      ovr_val=None, drop_cols=None, fit_params=None):
        '''
        Wrapper for testing hyper-parameter optimization models with the
        OrderedOVRClassifier API against a (possibly) classification-masked X
        dataset.

        Note that if an OVR model has been attached to the pipeline, the same
        dataset(s) used to train/evaluate the first OVR model must be used to
        train future OrderedOVRClassifier pipeline steps.

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

        eval_set: DataFrame or list of (X, y) tuple, optional
            Dataset to use as validation set for early-stopping and/or scoring
            trained models.

        ovr_val: str, int, or float, optional
            If specified, fit_test_grid will perform OVR modeling against the
            ovr_val classification label.

        drop_cols: list of str, optional
            Labels of columns to ignore in modeling, only applicable to pandas
            DataFrame X input.

        fit_params: dict, optional
            Key-value pairs of optional arguments to pass into model fit
            function.

        Returns
        -------
        grid_model: GridSearchCV or RandomizedSearchCV model
            Hyper-parameter optimizer model with recorded optimization results.
            Note that by design, retrain is set to False, and the user will
            need to train a new model with the best parameters found if they
            choose to attach the model to the OrderedOVRClassifier pipeline.
        '''
        mclass = str(grid_model.estimator.__class__).split("'")[-2].split('.')

        # checks if grid_model is a Pipeline class
        if mclass[-1][:8] == 'Pipeline':
            grid_model.estimator = u.PipelineES(grid_model.estimator.steps,
                                                grid_model.estimator.memory)
            prefix = grid_model.estimator.steps[-1][0] + '__'
            estimator = grid_model.estimator.steps[-1][1]
        else:
            prefix = None
            estimator = grid_model.estimator

        if estimator.__module__ == 'lightgbm.sklearn':
            lgbm_grid = True
        else:
            lgbm_grid = False

        X, y, drop_cols = self._xy_transform(X, y, drop_cols)
        eval_set = self._eval_set(eval_set, drop_cols)

        if ovr_val is not None:
            assert ovr_val in y.unique()
            key = ovr_val
        else:
            key = 'final'

        Xm, ym, eval_X, eval_y = self._mask_datasets(X, y, eval_set, ovr_val)

        if lgbm_grid:
            ym, eval_y, _ = self._encode_y(ym, eval_y, lgbm_grid)

        _, fit_params = self._get_model(key, eval_X, eval_y, model=estimator,
                                        fit_params=fit_params, prefix=prefix)

        if eval_X is not None:
            n = 1 + int(mclass[-1][:8] == 'Pipeline')

            if X.__class__ == pd.DataFrame:
                X = pd.concat([Xm] + list(repeat(eval_X, n)))
            elif X.__class__ == np.ndarray:
                X = np.vstack([Xm] + list(repeat(eval_X, n)))

            y = np.hstack([ym] + list(repeat(eval_y, n)))

            end_train = len(y) - len(eval_y)
            end_test = len(y)
            cv = [(np.arange(end_train), np.arange(end_train, end_test))]

            if n == 2:
                end_train = len(y) - 2*len(eval_y)
                end_test = len(y) - len(eval_y)

                eval_idx = [(np.arange(end_train),
                             np.arange(end_train, end_test))]
                fit_params['eval_idx'] = eval_idx

            grid_model.set_params(cv=cv, refit=False)

        grid_model = self._fit_model(grid_model, X, y, key, fit_params)

        prt = 'best_params:\n{}\n'.format(grid_model.best_params_)
        print('', '-'*80, prt, sep='\n')

        if type(grid_model.scoring) == str:
            print('best {0} score:\n{1}'.format(grid_model.scoring,
                                                grid_model.best_score_))
        else:
            print('best score:\n{}'.format(grid_model.best_score_))

        print('-'*80, '', sep='\n')

        return grid_model

    def fit_test_ovr(self, model, ovr_val, X, y=None, eval_set=None,
                     drop_cols=None, fbeta_weight=1.0, fit_params=None):
        '''
        Function for training an OVR model against a (possibly) classification
        masked X dataset. Does not attach trained model to the pipeline for
        OrderedOVRClassifier. Also evaluates binary classification with the
        imported plot_thresholds function, which plots precision, recall, and
        fscores for all thresholds with 0.01 interval spacing.

        Note that if an OVR model has been attached to the pipeline, the same
        dataset(s) used to train/evaluate the first OVR model must be used to
        train future OrderedOVRClassifier pipeline steps.

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

        eval_set: DataFrame or list of (X, y) tuple, optional
            Dataset to use as validation set for early-stopping and/or scoring
            trained models.

        drop_cols: list of str, optional
            Labels of columns to ignore in modeling, only applicable to pandas
            DataFrame X input.

        fbeta_weight: float, optional, default: 1.0
            The strength of recall versus precision in the F-score.

        fit_params: dict, optional
            Key-value pairs of optional arguments to pass into model fit
            function.

        Returns
        -------
        model: OOVR_Model
            OVR fitted model trained against classification-masked X dataset.
        '''
        X, y, drop_cols = self._xy_transform(X, y, drop_cols)
        eval_set = self._eval_set(eval_set, drop_cols)

        assert ovr_val in y.unique()

        # fit LabelEncoder if not fitted yet, otherwise take existing
        if hasattr(self._le, 'classes_'):
            enc = self._le
        else:
            enc = self._le.fit(y)

        model = self._fit_ovr(X, y, eval_set, [ovr_val], fbeta_weight, enc,
                              attach=False, model=model, fit_params=fit_params)

        return model

    def multiclassification_report(self, X, y=None, drop_cols=None):
        '''
        Wrapper function for extended_classification_report, which is an
        extension of sklearn.metrics.classification_report. Builds a text
        report showing the main classification metrics and the total count of
        multiclass predictions per class.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Data used for predictions.

        y: array-like, shape = [n_samples, ], optional
            True labels for X. If not provided and X is a DataFrame, will
            extract y column from X with the provided self.target value.

        drop_cols: list of str, optional
            Labels of columns ignored in modeling, only applicable to pandas
            DataFrame X input.
        '''
        X, y, _ = self._xy_transform(X, y, drop_cols)
        y_pred = self.predict(X)

        return u.extended_classification_report(y, y_pred)

    def plot_feature_importance(self, X, y=None, filter_class=None, n_jobs=-1,
                                n_samples=5000, progressbar=True,
                                drop_cols=None):
        '''
        Wrapper function for calling the plot_feature_importance function from
        skater, which estimates the feature importance of all columns based on
        a random sample of 5000 data points. To calculate feature importance
        the following procedure is executed:

        1. Calculate the original probability predictions for each class.
        2. Loop over the columns, one at a time, repeating steps 3-5 each time.
        3. Replace the entire column corresponding to the variable of interest
           with replacement values randomly sampled from the column of interest
        4. Use the model to predict the probabilities.
        5. The (column, average_absolute_probability_difference) becomes an
           (x, y) pair of the feature importance plot.
        6. Normalize the average_probability_difference so the sum equals 1.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Input data used for training or evaluating the fitted model.

        y: array-like, shape = [n_samples, ], optional
            True labels for X. If not provided and X is a DataFrame, will
            extract y column from X with the provided self.target value.

        filter_class: str or numeric, optional
            If specified, the feature importances will only be calculated for y
            data points matching class specified for filter_class.

        n_jobs: int, optional, default: -1
            The number of CPUs to use to compute the feature importances. -1
            means 'all CPUs' (default).

        n_samples: int, optional, default: 5000
            How many samples to use when computing importance.

        progressbar: bool, optional, default: True
            Whether to display progress. This affects which function we use to
            multipool the function execution, where including the progress bar
            results in 10-20% slowdowns.

        drop_cols: list of str, optional
            Labels of columns ignored in modeling, only applicable to pandas
            DataFrame X input.
        '''
        X, y, _ = self._xy_transform(X, y, drop_cols)
        X, n_jobs, col_names = self._skater_extract(X, n_jobs)

        return u.plot_feature_importance(self, X, y, col_names, filter_class,
                                         n_jobs, n_samples, progressbar)

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
            Labels of columns ignored in modeling, only applicable to pandas
            DataFrame X input.
        '''
        X, y, _ = self._xy_transform(X, y, drop_cols)
        return u.plot_oovr_dependencies(self, ovr_val, X, y, comp_vals)

    def plot_partial_dependence(self, X, col, grid_resolution=100,
                                grid_range=(.05, 0.95), n_jobs=-1,
                                n_samples=1000, progressbar=True,
                                drop_cols=None):
        '''
        Wrapper function for calling the plot_partial_dependence function from
        skater, which estimates the partial dependence of a column based on a
        random sample of 1000 data points. To calculate partial dependencies
        the following procedure is executed:

        1. Pick a range of values (decided by the grid_resolution and
           grid_range parameters) to calculate partial dependency for.
        2. Loop over the values, one at a time, repeating steps 3-5 each time.
        3. Replace the entire column corresponding to the variable of interest
           with the current value that is being cycled over.
        4. Use the model to predict the probabilities.
        5. The (value, average_probability) becomes an (x, y) pair of the
           partial dependence plot.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Input data used for training or evaluating the fitted model.

        col: str
            Label for the feature to compute partial dependence for.

        grid_resolution: int, optional, default: 100
            How many unique values to include in the grid. If the percentile
            range is 5% to 95%, then that range will be cut into
            <grid_resolution> equally size bins.

        grid_range: (float, float) tuple, optional, default: (.05, 0.95)
            The percentile extrama to consider. 2 element tuple, increasing,
            bounded between 0 and 1.

        n_jobs: int, optional, default: -1
            The number of CPUs to use to compute the partial dependence. -1
            means 'all CPUs' (default).

        n_samples: int, optional, default: 1000
            How many samples to use when computing partial dependence.

        progressbar: bool, optional, default: True
            Whether to display progress. This affects which function we use to
            multipool the function execution, where including the progress bar
            results in 10-20% slowdowns.

        drop_cols: list of str, optional
            Labels of columns ignored in modeling, only applicable to pandas
            DataFrame X input.
        '''
        X = self._pred_cleanup(X, drop_cols)
        X, n_jobs, col_names = self._skater_extract(X, n_jobs)

        return u.plot_2d_partial_dependence(self, X, col, col_names,
                                            grid_resolution, grid_range,
                                            n_jobs, n_samples, progressbar)

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
            Labels of columns ignored in modeling, only applicable to pandas
            DataFrame X input.

        Returns
        -------
        pred: array-like, shape = [n_samples, ]
            Predicted multi-class targets.
        '''
        X = self._pred_cleanup(X, drop_cols)
        pred = np.array(np.zeros(len(X)), self._le.classes_.dtype)
        mask = np.zeros(len(X)).astype(bool)

        # predictions from the One-Vs-Rest classifiers
        for i in np.arange(len(self.ovr_vals))[start:]:
            clf = self.pipeline[i][1]
            self._xg_cleanup(clf)

            thld = self.thresholds[i]
            pred_clf = clf.predict_proba(X[~mask])[:, 1] >= thld
            pred[np.where(~mask)[0][pred_clf]] = self.ovr_vals[i]

            mask[~mask] = np.logical_or(mask[~mask], pred_clf)

            if all(mask):  # all classifications have been made
                return pred

        clf_final = self.pipeline[-1][1]
        self._xg_cleanup(clf_final)

        # predictions from the final classifier
        pred_final = clf_final.predict(X[~mask])
        pred_final = self._le.inverse_transform(pred_final)
        pred[~mask] = pred_final

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
        pred = self.predict(row)[0]
        return pred

    def predict_proba(self, X, score_type='uniform', drop_cols=None):
        '''
        Predict probabilities for multi-class targets using underlying
        estimators. Because each classifier is trained against different
        classes in Ordered One-Vs-Rest modeling, it is not possible to output
        accurate probabilities that always return the correct prediction (from
        the predict function) for the most probable class. Instead, the
        following score_type methods are used to output probability estimates.

        If the score_type is 'raw', the probability score from the specific
        model used to train the class of interest is returned for each class.
        There are no corrections applied for the 'raw' score_type and the
        outputted probabilities will not sum to 1.

        If the score_type is 'chained', the probability of the next classifier
        in the pipeline is scaled down so the probabilities sum to the negative
        ('rest') classification probability of the current classifier.

        If the score type is 'uniform', positive values for Ordered One-Vs_Rest
        classifications are treated in the same manner as the 'chained'
        score_type. Negative ('rest') outcomes always return a uniform value
        based on the 1-precision score for the 'rest' class of the binary model
        used in the pipeline step for the One-Vs-Rest classifier. This ensures
        that future pipeline models that sub-classify the 'rest' classification
        will always sum up to the same number, allowing more meaningful
        interpretation of the probabilities.

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_features]
            Data used for predictions.

        score_type: str, optional, default: 'uniform'
            Acceptable inputs are 'raw', 'chained', and 'uniform'.

        drop_cols: list of str, optional
            Labels of columns ignored in modeling, only applicable to pandas
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

        # Initiate new array
        pred = np.zeros((n, m))
        proba_cols = np.setdiff1d(np.arange(m), ovr_cols)
        pred[:, proba_cols] = probas

        # Set probabilities
        for i in np.arange(len(self.ovr_vals))[::-1]:
            clf = self.pipeline[i][1]
            self._xg_cleanup(clf)
            ovr_proba = clf.predict_proba(X)[:, -1]

            if score_type == 'raw':
                pred[:, ovr_cols[i]] = ovr_proba

            elif score_type == 'chained' or self._all_binary:
                pred *= (1 - ovr_proba)[:, np.newaxis]
                pred[:, ovr_cols[i]] = ovr_proba

            elif score_type == 'uniform':
                thld = self.thresholds[i]
                guess_prob = 1 - self.rest_precision[i]
                mask = ovr_proba >= thld

                # chained predictions for positive results
                pred[mask] *= (1 - ovr_proba[mask])[:, np.newaxis]
                pred[mask, ovr_cols[i]] = ovr_proba[mask]

                # use uniform probability for negative results
                pred[~mask, :] = pred[~mask, :] * (1-guess_prob)
                pred[~mask, ovr_cols[i]] = guess_prob

            else:
                raise RuntimeError('Bad input for score_type.')

        return pred

    def predict_proba_json(self, row, score_type='uniform', print_prob=False):
        '''
        Predict probabilities for multi-class target from JSON using
        underlying estimators. Because each classifier is trained against
        different classes in Ordered One-Vs-Rest modeling, it is not possible
        to output accurate probabilities that always return the correct
        prediction for the most probable class. Instead, the following
        score_type methods are used to output probability estimates.

        If the score_type is 'raw', the probability score from the specific
        model used to train the class of interest is returned for each class.
        There are no corrections applied for the 'raw' score_type and the
        outputted probabilities will not sum to 1.

        If the score_type is 'chained', the probability of the next classifier
        in the pipeline is scaled down so the probabilities sum to the negative
        ('rest') classification probability of the current classifier.

        If the score type is 'uniform', positive values for Ordered One-Vs_Rest
        classifications are treated in the same manner as the 'chained'
        score_type. Negative ('rest') outcomes always return a uniform value
        based on the 1-precision score for the 'rest' class of the binary model
        used in the pipeline step for the One-Vs-Rest classifier. This ensures
        that future pipeline models that subclassify the 'rest' classification
        will always sum up to the same number, allowing more meaniningful
        interpretation of the probabilities.

        Parameters
        ----------
        row: json
            Single JSON row to make prediction from.

        score_type: str, optional, default: 'uniform'
            Acceptable inputs are 'raw', 'chained', and 'uniform'.

        print_prob: bool, optional
            Whether to print out the probabilities to console.

        Returns
        -------
        pred: array-like, shape = [1, n_classes] or None
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in self._le.classes_ or
            returns None if print_prob is True.
        '''
        row = self._json_transform(row)
        pred = self.predict_proba(row, score_type=score_type)[0]

        if print_prob:
            ljust = max([len(str(x)) for x in self._le.classes_]) + 1

            if score_type == 'raw':
                m = len(self._le.classes_)
                ovr_cols = self._le.transform(self.ovr_vals)
                proba_cols = np.setdiff1d(np.arange(m), ovr_cols)

                # print out the raw probabilities for each classifier in the
                # same order as the modeling pipeline
                for i, c in enumerate(ovr_cols):
                    label = str(self._le.classes_[c])
                    prt1 = 'OVR Classifier, Class {}'.format(self.ovr_vals[i])
                    prt2 = '{}: {:.3f}'.format(label.ljust(ljust), pred[c])
                    print(prt1, '-'*len(prt1), prt2, '', sep='\n')

                prt = 'Classifier for Remaining Classes:'
                print(prt, '-'*len(prt), sep='\n')

                for i, c in enumerate(proba_cols):
                    label = str(self._le.classes_[c])
                    print('{}: {:.3f}'.format(label.ljust(ljust), pred[c]))

            else:
                # print out probabilities in ascening class label order
                for i, p in enumerate(pred):
                    label = str(self._le.classes_[i])
                    print('{}: {:.3f}'.format(label.ljust(ljust), p))

            return

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
            Labels of columns ignored in modeling, only applicable to pandas
            DataFrame X input.

        Returns
        -------
        scr: float
            Mean accuracy of self.predict(X) wrt y.
        '''
        X, y, _ = self._xy_transform(X, y, drop_cols)
        scr = m.accuracy_score(y, self.predict(X), sample_weight=sample_weight)

        return scr
