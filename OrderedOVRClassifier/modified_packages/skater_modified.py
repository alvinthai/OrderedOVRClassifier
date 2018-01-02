"""
Partial Dependence class

Adopted from https://github.com/datascienceinc/Skater
Code has been modified to disable multiprocessing for predictions from LightGBM

The patched code is compatible with skater version 1.0.3
"""

from skater.core.global_interpretation.partial_dependence import PartialDependence

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


class PartialDependence(PartialDependence):
    """Contains methods for partial dependence. Subclass of BaseGlobalInterpretation

       Partial dependence adapted from:

       T. Hastie, R. Tibshirani and J. Friedman,
       Elements of Statistical Learning Ed. 2, Springer, 2009.
    """
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
