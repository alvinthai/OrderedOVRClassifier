"""
Partial Dependence class

Adopted from https://github.com/datascienceinc/Skater
Code has been modified to disable multiprocessing for predictions from LightGBM

The patched code is compatible with skater version 1.0.3
"""


from itertools import product, cycle
import numpy as np
import pandas as pd
from multiprocess import Pool
import functools
import logging

from skater.data import DataManager
from skater.core.global_interpretation.base import BaseGlobalInterpretation
from skater.core.global_interpretation.partial_dependence import _compute_pd
from skater.model.base import ModelType
from skater.util import exceptions
from skater.util.user_defined_types import ControlledDict
from skater.util.kernels import flatten
from skater.util.plotting import COLORS
from skater.util.exceptions import *
from skater.util.static_types import StaticTypes
from skater.util.progressbar import ProgressBar


# if we want to employ instance methods in multiprocessing, enable this code:
# copy_reg.pickle(types.MethodType, pickle_method, unpickle_method)
# methods stored in util.serialization


class PartialDependence(BaseGlobalInterpretation):
    """Contains methods for partial dependence. Subclass of BaseGlobalInterpretation

       Partial dependence adapted from:

       T. Hastie, R. Tibshirani and J. Friedman,
       Elements of Statistical Learning Ed. 2, Springer, 2009.
    """

    __all__ = ['partial_dependence', 'plot_partial_dependence']

    _sd_names_ = {'prediction': 'sd_prediction', 'estimate': 'sd_estimate'}

    def _build_metadata_dict(self, modelinstance, pd_feature_ids, data_feature_ids, filter_classes, variance_type):

        feature_columns = [self.feature_column_name_formatter(i) for i in pd_feature_ids]
        sd_col = PartialDependence._sd_names_[variance_type]
        if filter_classes is not None:
            filtered_target_names = [i for i in modelinstance.target_names if i in filter_classes]
        else:
            filtered_target_names = None
        metadata = ControlledDict({
            'sd_column': sd_col,
            'target_names': modelinstance.target_names,
            'filtered_target_names': filtered_target_names,
            'feature_columns_for_pd': feature_columns,
            'feature_ids_for_pd': pd_feature_ids,
            'all_feature_ids': data_feature_ids,
        })
        metadata.block_setitem()
        return metadata

    @staticmethod
    def feature_column_name_formatter(columnname):
        return columnname

    def _check_features(self, feature_ids):
        if StaticTypes.data_types.is_string(feature_ids) or StaticTypes.data_types.is_numeric(feature_ids):
            feature_ids = [feature_ids]

        if len(feature_ids) >= 3:
            too_many_features_err_msg = "Pass in at most 2 features for pdp. If you have a \n" \
                                        "use case where you'd like to look at 3 simultaneously \n" \
                                        ", please let us know."
            raise(exceptions.TooManyFeaturesError(too_many_features_err_msg))

        if len(feature_ids) == 0:
            empty_features_err_msg = "Feature ids must have non-zero length"
            raise(exceptions.EmptyFeatureListError(empty_features_err_msg))

        if len(set(feature_ids)) != len(feature_ids):
            duplicate_features_error_msg = "feature_ids cannot contain duplicate values"
            raise(exceptions.DuplicateFeaturesError(duplicate_features_error_msg))

        return feature_ids

    def partial_dependence(self, feature_ids, modelinstance, filter_classes=None, grid=None,
                           grid_resolution=30, n_jobs=-1, grid_range=None, sample=True,
                           sampling_strategy='random-choice', n_samples=1000,
                           bin_count=50, return_metadata=False,
                           progressbar=True, variance_type='estimate'):

        """
        Approximates the partial dependence of the predict_fn with respect to the
        variables passed.

        Parameters:
        -----------
        feature_ids: list
            the names/ids of the features for which partial dependence is to be computed.
            Note that the algorithm's complexity scales exponentially with additional
            features, so generally one should only look at one or two features at a
            time. These feature ids must be available in the class's associated DataSet.
            As of now, we only support looking at 1 or 2 features at a time.
        modelinstance: skater.model.model.Model subtype
            an estimator function of a fitted model used to derive prediction. Supports
            classification and regression. Supports classification(binary, multi-class) and regression.
            predictions = predict_fn(data)

            Can either by a skater.model.remote.DeployedModel or a
            skater.model.local.InMemoryModel
        filter_classes: array type
            The classes to run partial dependence on. Default None invokes all classes.
            Only used in classification models.
        grid: numpy.ndarray
            2 dimensional array on which we fix values of features. Note this is
            determined automatically if not given based on the percentiles of the
            dataset.
        grid_resolution: int
            how many unique values to include in the grid. If the percentile range
            is 5% to 95%, then that range will be cut into <grid_resolution>
            equally size bins. Defaults to 30.
        n_jobs: int
            The number of CPUs to use to compute the PDs. -1 means 'all CPUs'.
            Defaults to using all cores(-1).
        grid_range: tuple
            the percentile extrama to consider. 2 element tuple, increasing, bounded
            between 0 and 1.
        sample: boolean
            Whether to sample from the original dataset.
        sampling_strategy: string
            If sampling, which approach to take. See DataSet.generate_sample for
            details.
        n_samples: int
            The number of samples to use from the original dataset. Note this is
            only active if sample = True and sampling strategy = 'uniform'. If
            using 'uniform-over-similarity-ranks', use samples per bin
        bin_count: int
            The number of bins to use when using the similarity based sampler. Note
            this is only active if sample = True and
            sampling_strategy = 'uniform-over-similarity-ranks'.
            total samples = bin_count * samples per bin.
        samples_per_bin: int
            The number of samples to collect for each bin within the sampler. Note
            this is only active if sample = True and
            sampling_strategy = 'uniform-over-similarity-ranks'. If using
            sampling_strategy = 'uniform', use n_samples.
            total samples = bin_count * samples per bin.
        variance_type: string

        return_metadata: boolean

        :Example:
        >>> from skater.model import InMemoryModel
        >>> from skater.core.explanations import Interpretation
        >>> from sklearn.ensemble import RandomForestClassier
        >>> from sklearn.datasets import load_boston
        >>> boston = load_boston()
        >>> X = boston.data
        >>> y = boston.target
        >>> features = boston.feature_names

        >>> rf = RandomForestClassier()
        >>> rf.fit(X,y)


        >>> model = InMemoryModel(rf, examples = X)
        >>> interpreter = Interpretation()
        >>> interpreter.load_data(X)
        >>> feature_ids = ['ZN','CRIM']
        >>> interpreter.partial_dependence.partial_dependence(features,model)
        """

        if self.data_set is None:
            load_data_not_called_err_msg = "self.interpreter.data_set not found. \n" \
                                           "Please call Interpretation.load_data \n" \
                                           "before running this method."
            raise(exceptions.DataSetNotLoadedError(load_data_not_called_err_msg))

        feature_ids = self._check_features(feature_ids)

        if filter_classes:
            err_msg = "members of filter classes must be \n" \
                      "members of modelinstance.classes. \n" \
                      "Expected members of: \n" \
                      "{0}\n" \
                      "got: \n" \
                      "{1}".format(modelinstance.target_names,
                                   filter_classes)
            filter_classes = list(filter_classes)
            assert all([i in modelinstance.target_names for i in filter_classes]), err_msg

        # TODO: There might be a better place to do this check
        if not isinstance(modelinstance, ModelType):
            raise(exceptions.ModelError("Incorrect estimator function used for computing partial dependence, try one \n"
                                        "creating one with skater.model.local.InMemoryModel or \n"
                                        "skater.model.remote.DeployedModel"))

        if modelinstance.model_type == 'classifier' and modelinstance.probability is False:

            if modelinstance.unique_values is None:
                raise(exceptions.ModelError('If using classifier without probability scores, unique_values cannot \n'
                                            'be None'))
            self.interpreter.logger.warn("Classifiers with probability scores can be explained \n"
                                         "more granularly than those without scores. If a prediction method with \n"
                                         "scores is available, use that instead.")

        # TODO: This we can change easily to functional style
        missing_feature_ids = []
        for feature_id in feature_ids:
            if feature_id not in self.data_set.feature_ids:
                missing_feature_ids.append(feature_id)

        if missing_feature_ids:
            missing_feature_id_err_msg = "Features {0} not found in \n" \
                                         "Interpretation.data_set.feature_ids \n" \
                                         "{1}".format(missing_feature_ids, self.data_set.feature_ids)
            raise(KeyError(missing_feature_id_err_msg))

        if grid_range is None:
            grid_range = (.05, 0.95)
        else:
            if not hasattr(grid_range, "__iter__"):
                err_msg = "Grid range {} needs to be an iterable".format(grid_range)
                raise(exceptions.MalformedGridRangeError(err_msg))

        self._check_grid_range(grid_range)

        if not modelinstance.has_metadata:
            examples = self.data_set.generate_sample(strategy='random-choice',
                                                     sample=True,
                                                     n_samples=10)

            examples = DataManager(examples, feature_names=self.data_set.feature_ids)
            modelinstance._build_model_metadata(examples)

        # if you dont pass a grid, build one.
        grid = np.array(grid)
        if not grid.any():
            # Currently, if a given feature has fewer unique values than the value
            # of grid resolution, then the grid will be set to those unique values.
            # Otherwise it will take the percentile
            # range according with grid_resolution bins.
            grid = self.data_set.generate_grid(feature_ids,
                                               grid_resolution=grid_resolution,
                                               grid_range=grid_range)
        else:
            # want to ensure all grids have 2 axes
            if len(grid.shape) == 1 and \
                    (StaticTypes.data_types.is_string(grid[0]) or StaticTypes.data_types.is_numeric(grid[0])):
                grid = grid[:, np.newaxis].T
                grid_resolution = grid.shape[1]

        self.interpreter.logger.debug("Grid shape used for pdp: {}".format(grid.shape))
        self.interpreter.logger.debug("Grid resolution for pdp: {}".format(grid_resolution))

        # make sure data_set module is giving us correct data structure
        self._check_grid(grid, feature_ids)

        # generate data
        data_sample = self.data_set.generate_sample(strategy=sampling_strategy,
                                                    sample=sample,
                                                    n_samples=n_samples,
                                                    bin_count=bin_count)

        assert type(data_sample) == self.data_set.data_type, "Something went wrong\n" \
                                                             "Theres a type mismatch between\n" \
                                                             "the sampled data and the origina\nl" \
                                                             "training set. Check Skater.models\n"

        _pdp_metadata = self._build_metadata_dict(modelinstance,
                                                  feature_ids,
                                                  self.data_set.feature_ids,
                                                  filter_classes,
                                                  variance_type)

        self.interpreter.logger.debug("Shape of sampled data: {}".format(data_sample.shape))
        self.interpreter.logger.debug("Feature Ids: {}".format(feature_ids))
        self.interpreter.logger.debug("PD metadata: {}".format(_pdp_metadata))

        # cartesian product of grid
        grid_expanded = pd.DataFrame(list(product(*grid))).values

        if grid_expanded.shape[0] <= 0:
            empty_grid_expanded_err_msg = "Must have at least 1 pdp value" \
                                          "grid shape: {}".format(grid_expanded.shape)
            raise(exceptions.MalformedGridError(empty_grid_expanded_err_msg))

        predict_fn = modelinstance._get_static_predictor()

        n_jobs = None if n_jobs < 0 else n_jobs
        pd_func = functools.partial(_compute_pd,
                                    estimator_fn=predict_fn,
                                    grid_expanded=grid_expanded,
                                    pd_metadata=_pdp_metadata,
                                    input_data=data_sample,
                                    filter_classes=filter_classes)
        arg_list = [i for i in range(grid_expanded.shape[0])]

        executor_instance = Pool(n_jobs)

        if progressbar:
            self.interpreter.logger.warn("Progress bars slow down runs by 10-20%. For slightly "
                                         "faster runs, do progressbar=False")
            mapper = executor_instance.imap
            p = ProgressBar(len(arg_list), units='grid cells')
        else:
            mapper = executor_instance.map

        pd_list = []
        try:
            if n_jobs == 1:
                raise ValueError("Skipping to single processing")
            for pd_row in mapper(pd_func, arg_list):
                if progressbar:
                    p.animate()
                pd_list.append(pd_row)
        except:
            self.interpreter.logger.info("Multiprocessing failed, going single process")
            for pd_row in map(pd_func, arg_list):
                if progressbar:
                    p.animate()
                pd_list.append(pd_row)
        finally:
            executor_instance.close()
            executor_instance.join()
            executor_instance.terminate()

        if return_metadata:
            return pd.DataFrame(list(pd_list)), _pdp_metadata
        else:
            return pd.DataFrame(list(pd_list))

    def plot_partial_dependence(self, feature_ids, modelinstance, filter_classes=None,
                                grid=None, grid_resolution=30, grid_range=None,
                                n_jobs=-1, sample=True, sampling_strategy='random-choice',
                                n_samples=1000, bin_count=50, with_variance=False,
                                figsize=(6, 4), progressbar=True, variance_type='estimate'):
        """
        Computes partial_dependence of a set of variables. Essentially approximates
        the partial partial_dependence of the predict_fn with respect to the variables
        passed.

        Parameters:
        -----------
        feature_ids: list
            the names/ids of the features for which partial dependence is to be computed.
            Note that the algorithm's complexity scales exponentially with additional
            features, so generally one should only look at one or two features at a
            time. These feature ids must be available in the class's associated DataSet.
            As of now, we only support looking at 1 or 2 features at a time.
        modelinstance: skater.model.model.Model subtype
            an estimator function of a fitted model used to derive prediction. Supports
            classification and regression. Supports classification(binary, multi-class) and regression.
            predictions = predict_fn(data)

            Can either by a skater.model.remote.DeployedModel or a
            skater.model.local.InMemoryModel
        grid: numpy.ndarray
            2 dimensional array on which we fix values of features. Note this is
            determined automatically if not given based on the percentiles of the
            dataset.
        grid_resolution: int
            how many unique values to include in the grid. If the percentile range
            is 5% to 95%, then that range will be cut into <grid_resolution>
            equally size bins. Defaults to 30.
        grid_range: tuple
            the percentile extrama to consider. 2 element tuple, increasing, bounded
            between 0 and 1.
        n_jobs: int
            The number of CPUs to use to compute the PDs. -1 means 'all CPUs'.
            Defaults to using all cores(-1).
        sample: boolean
            Whether to sample from the original dataset.
        sampling_strategy: string
            If sampling, which approach to take. See DataSet.generate_sample for
            details.
        n_samples: int
            The number of samples to use from the original dataset. Note this is
            only active if sample = True and sampling strategy = 'uniform'. If
            using 'uniform-over-similarity-ranks', use samples per bin
        bin_count: int
            The number of bins to use when using the similarity based sampler. Note
            this is only active if sample = True and
            sampling_strategy = 'uniform-over-similarity-ranks'.
            total samples = bin_count * samples per bin.
        samples_per_bin: int
            The number of samples to collect for each bin within the sampler. Note
            this is only active if sample = True and
            sampling_strategy = 'uniform-over-similarity-ranks'. If using
            sampling_strategy = 'uniform', use n_samples.
            total samples = bin_count * samples per bin.


        with_variance(bool):
            whether to include pdp error bars in the plots. Currently disabled for 3D
            plots for visibility. If you have a use case where you'd like error bars for
            3D pdp plots, let us know!
        plot_title(string):
            title for pdp plots
        variance_type: string
            if variance plotting is enabled, determines which variance to include.
            estimate: the variance of the partial dependence estimates
            prediction: the variances of the predictions at the given point

        Example
        --------
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> from sklearn.datasets.california_housing import fetch_california_housing
        >>> cal_housing = fetch_california_housing()
        # split 80/20 train-test
        >>> x_train, x_test, y_train, y_test = train_test_split(cal_housing.data,
        >>>                             cal_housing.target, test_size=0.2, random_state=1)
        >>> names = cal_housing.feature_names
        >>> print("Training the estimator...")
        >>> estimator = GradientBoostingRegressor(n_estimators=10, max_depth=4,
        >>>                             learning_rate=0.1, loss='huber', random_state=1)
        >>> estimator.fit(x_train, y_train)
        >>> from skater.core.explanations import Interpretation
        >>> interpreter = Interpretation()
        >>> print("Feature name: {}".format(names))
        >>> interpreter.load_data(X_train, feature_names=names)
        >>> print("Input feature name: {}".format[names[1], names[5]])
        >>> from skater.model import InMemoryModel
        >>> model = InMemoryModel(clf.predict, examples = X_train)
        >>> interpreter.partial_dependence.plot_partial_dependence([names[1], names[5]], model,
        >>>                                                         n_samples=100, n_jobs=1)

        """

        global pyplot
        global ScalarFormatter
        global Axes3D
        global mpl_axes
        global cm
        global tick_formatter
        from matplotlib.axes._subplots import Axes as mpl_axes
        # from matplotlib.ticker import ScalarFormatter
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import pyplot, cm
        from skater.util.plotting import tick_formatter

        # in the event that a user wants a 3D pdp with multiple classes, how should
        # we handle this? currently each class will get its own figure
        if not hasattr(feature_ids, "__iter__"):
            pd_df, metadata = self.partial_dependence(feature_ids, modelinstance,
                                                      filter_classes=filter_classes, grid=grid,
                                                      grid_resolution=grid_resolution,
                                                      grid_range=grid_range, sample=sample,
                                                      sampling_strategy=sampling_strategy,
                                                      n_samples=n_samples, bin_count=bin_count,
                                                      n_jobs=n_jobs, return_metadata=True,
                                                      progressbar=progressbar,
                                                      variance_type=variance_type)

            self.interpreter.logger.info("done computing pd, now plotting ...")
            ax = self._plot_pdp_from_df(pd_df, metadata, with_variance=with_variance, figsize=figsize)
            return ax
        else:
            ax_list = []
            for feature_or_feature_pair in feature_ids:
                pd_df, metadata = self.partial_dependence(feature_or_feature_pair, modelinstance,
                                                          filter_classes=filter_classes, grid=grid,
                                                          grid_resolution=grid_resolution,
                                                          grid_range=grid_range, sample=sample,
                                                          sampling_strategy=sampling_strategy,
                                                          n_samples=n_samples, bin_count=bin_count,
                                                          n_jobs=n_jobs, return_metadata=True,
                                                          progressbar=progressbar,
                                                          variance_type=variance_type)

                self.interpreter.logger.info("done computing pd, now plotting ...")
                ax = self._plot_pdp_from_df(pd_df, metadata, with_variance=with_variance, figsize=figsize)
                ax_list.append(ax)
            return ax_list

    def _plot_pdp_from_df(self, pdp, pd_metadata,
                          with_variance=False, plot_title=None,
                          disable_offset=True, figsize=(16, 10)):

        feature_columns = pd_metadata['feature_columns_for_pd']
        if pd_metadata['filtered_target_names'] is None:
            target_columns = pd_metadata['target_names']
        else:
            target_columns = pd_metadata['filtered_target_names']
        sd_col = pd_metadata['sd_column']
        n_features = len(feature_columns)
        if n_features == 1 or not hasattr(feature_columns, "__iter__"):
            feature_column = feature_columns[0]
            return self._2d_pdp_plot(pdp,
                                     feature_column,
                                     sd_col,
                                     target_columns,
                                     with_variance=with_variance,
                                     plot_title=plot_title,
                                     disable_offset=disable_offset,
                                     figsize=figsize)
        else:
            msg = "Something went wrong. Expected either a single feature, " \
                  "got array of size:" \
                  "{}: {}".format(*[n_features, feature_columns])
            raise(ValueError(msg))

    def _2d_pdp_plot(self, pdp, feature_name, sd_col, target_columns,
                     with_variance=False, plot_title=None,
                     disable_offset=True, figsize=(6, 4)):
        colors = cycle(COLORS)
        figure_list, axis_list = [], []

        # if there are just 2 classes, pick the last one.
        if len(target_columns) == 2:
            target_columns = [target_columns[-1]]

        for target_column in target_columns:
            f, ax = pyplot.subplots(1, figsize=figsize)
            figure_list.append(f)
            axis_list.append(ax)
            color = next(colors)

            data = pdp.set_index(feature_name)
            plane = data[target_column]

            # if binary feature, then len(pdp) == 2 -> barchart
            if self._is_feature_binary(pdp, feature_name) or not self.data_set.feature_info[feature_name]['numeric']:
                if with_variance:
                    error = data[sd_col]
                else:
                    error = None
                plane.plot(kind='bar', ax=ax, color=color, yerr=error)
            else:
                plane.plot(ax=ax, color=color)
                if with_variance:
                    upper_plane = plane + data[sd_col]
                    lower_plane = plane - data[sd_col]
                    ax.fill_between(data.index.values,
                                    lower_plane.values,
                                    upper_plane.values,
                                    alpha=.2,
                                    color=color)
            if plot_title:
                ax.set_title(plot_title)
            ax.set_ylabel(target_column)
            ax.set_xlabel(feature_name)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
            if disable_offset:
                ax.yaxis.set_major_formatter(tick_formatter())
        return flatten([figure_list, axis_list])

    def _is_feature_binary(self, pdp, feature):
        data = pdp[feature].values
        if len(np.unique(data)) == 2:
            return True
        else:
            return False

    @staticmethod
    def _check_grid(grid, feature_ids):
        if not isinstance(grid, np.ndarray):
            err_msg = "Grid of type {} must be a numpy array".format(type(grid))
            raise(exceptions.MalformedGridError(err_msg))

        if len(feature_ids) != grid.shape[0]:
            err_msg = "Given {0} features, there must be {1} rows in grid" \
                      "but {2} were found".format(len(feature_ids),
                                                  len(feature_ids),
                                                  grid.shape[0])
            raise(exceptions.MalformedGridError(err_msg))

    @staticmethod
    def _check_dataset(dataset):
        """
        Ensures that dataset is pandas dataframe, and dataset is not empty
        :param dataset: skater.__datatypes__
        :return:
        """
        if not isinstance(dataset, (pd.DataFrame, np.ndarray)):
            err_msg = "Dataset.data must be a pandas.DataFrame or numpy.ndarray"
            raise(exceptions.DataSetError(err_msg))

        if len(dataset) == 0:
            err_msg = "Dataset.data is empty"
            raise (exceptions.DataSetError(err_msg))

    @staticmethod
    def _check_grid_range(grid_range):
        """
        Tested by unit test, ensures grid range is between 0 and 1
        :param grid_range (tuple)

        """
        if len(grid_range) != 2:
            err_msg = "Grid range {} must have 2 elements".format(grid_range)
            raise(exceptions.MalformedGridRangeError(err_msg))
        if not all([i >= 0 and i <= 1 for i in grid_range]):
            err_msg = "All elements of grid range {} " \
                      "must be between 0 and 1".format(grid_range)
            raise(exceptions.MalformedGridRangeError(err_msg))
