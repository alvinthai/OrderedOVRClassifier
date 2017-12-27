API Reference
=============

.. _`[View source on GitHub]`: https://github.com/alvinthai/OrderedOVRClassifier/blob/master/OrderedOVRClassifier/classifier.py

`[View source on GitHub]`_

.. py:class:: OrderedOVRClassifier(target=None, ovr_vals=None, model_dict=None, model_fit_params=None, fbeta_weight=1.0, train_final_model=True, train_final_only=False)

  **Description**

    OrderedOVRClassifier is a custom scikit-learn module for approaching multi-classification with an Ordered One-Vs-Rest Modeling approach. Ordered One-Vs-Rest Classification performs a series of One-Vs-Rest Classifications where negative results are moved into subsequent training with previous classifications filtered out.

    The API for OrderedOVRClassifier is designed to be user-friendly with pandas, numpy, and scikit-learn. There is built in functionality to support easy handling for early stopping on the sklearn wrapper for XGBoost and LightGBM. If working with DataFrames, fitting a model with early stopping could be done using commands as simple as:

    .. code-block:: python

       oovr = OrderedOVRClassifier(target='label')
       oovr.fit(X=train_df, eval_set=eval_df)

    .. _notebook: http://nbviewer.jupyter.org/github/alvinthai/OrderedOVRClassifier/blob/master/examples/example.ipynb

    Refer to this notebook_ for a tutorial on how to use the API for OrderedOVRClassifier.

    OrderedOVRClassifier runs custom evaluation functions to diagnose and/or plot the predictive performance of the classification after training each model. With Ordered One-Vs-Rest Classification, the binary outcome from an Ordered One-Vs-Rest model can be optimized to achieve an ideal mix of accuracy/precision/recall scores among each predictive class. Call the :class:`plot_oovr_dependencies` function on a fully trained OrderedOVRClassifier model to execute these evaluations.

    OrderedOVRClassifier is designed to be modular and models can be tested without changing the fit state of OrderedOVRClassifier. These models can be manually attached to OrderedOVRClassifier at a later time. Additionally, a grid search wrapper is built into the API for hyper-parameter tuning against classification-subsetted datasets.

    OrderedOVRClassifier also includes utilities for model agnostic evaluation of feature importances and partial dependence. These model agnostic evaluation utilities (:class:`plot_feature_importance` and :class:`plot_partial_dependence`) require the skater library and are approximations based on a random sample of the data.

  **Parameters**

      target: str
          Label for target variable in pandas DataFrame. If provided, all future future inputs with an **X** DataFrame do not require an accompanying **y** input, as **y** will be extracted from the **X** DataFrame. However, the target column must be included in the **X** DataFrame for all fitting steps if the target parameter is provided.

      ovr_vals: list
          List of target values (and ordering) to perform ordered one-vs-rest.

      model_dict: dict of models
          Dictionary of models to perform ordered one-vs-rest, dict should include a model for each value in **ovr_vals**, and if train_final_model=True, a model specified for ``'final'``.

          .. code-block:: python

              model_dict = { value1 : LogisticRegression(),
                             value2 : RandomForestClassifier(),
                            'final' : XGBClassifier()}

      model_fit_params: dict of dict
          Additional parameters (inputted as a dict) to pass to the fit step of the models specified in model_dict.

          .. code-block:: python

              model_fit_params = { value1 : {'sample_weight': None},
                                   value2 : {'sample_weight': None},
                                  'final' : {'verbose': False}}

      fbeta_weight: float, default: 1.0
          The strength of recall versus precision in the F-score.

      train_final_model: bool, default: True
          Whether to train a final model to the remaining data after OVR fits.

      train_final_only: bool, default: False
          Whether to ignore OVR modeling and to train the final model only.

  **Methods**

      +----------------------------------------------------------------------------------------------------------------------+
      | **Core API**                                                                                                         |
      +----------------------------------------------------------------------------------------------------------------------+
      | :class:`fit` (X[, y, eval_Set, drop_cols])                                                                           |
      +----------------------------------------------------------------------------------------------------------------------+
      | :class:`predict` (X[, start, drop_cols])                                                                             |
      +----------------------------------------------------------------------------------------------------------------------+
      | :class:`predict_proba` (X[, score_type, drop_cols])                                                                  |
      +----------------------------------------------------------------------------------------------------------------------+
      | **Plotting API**                                                                                                     |
      +----------------------------------------------------------------------------------------------------------------------+
      | :class:`plot_feature_importance` (X[, y, filter_class, n_jobs, n_samples, progressbar, drop_cols])                   |
      +----------------------------------------------------------------------------------------------------------------------+
      | :class:`plot_partial_dependence` (X, col[, grid_resolution, grid_range, n_jobs, n_samples, progressbar, drop_cols])  |
      +----------------------------------------------------------------------------------------------------------------------+
      | :class:`plot_oovr_dependencies` (ovr_val, X[, y, comp_vals, drop_cols])                                              |
      +----------------------------------------------------------------------------------------------------------------------+
      | **Model Selection API**                                                                                              |
      +----------------------------------------------------------------------------------------------------------------------+
      | :class:`fit_test` (model, X[, y, eval_set, drop_cols])                                                               |
      +----------------------------------------------------------------------------------------------------------------------+
      | :class:`fit_test_ovr` (model, ovr_val, X[, y, eval_set, drop_cols])                                                  |
      +----------------------------------------------------------------------------------------------------------------------+
      | :class:`fit_test_grid` (grid_model, X[, y, eval_set, ovr_val, drop_cols])                                            |
      +----------------------------------------------------------------------------------------------------------------------+
      | :class:`attach_model` (oovr_model)                                                                                   |
      +----------------------------------------------------------------------------------------------------------------------+
      | **Miscellaneous API**                                                                                                |
      +----------------------------------------------------------------------------------------------------------------------+
      | :class:`multiclassification_report` (X[, y, drop_cols])                                                              |
      +----------------------------------------------------------------------------------------------------------------------+
      | :class:`predict_json` (row)                                                                                          |
      +----------------------------------------------------------------------------------------------------------------------+
      | :class:`predict_proba_json` (row[, score_type, print_prob])                                                          |
      +----------------------------------------------------------------------------------------------------------------------+
      | :class:`score` (X[, y, sample_weight, drop_cols])                                                                    |
      +----------------------------------------------------------------------------------------------------------------------+

Core API
--------

.. py:method:: OrderedOVRClassifier.fit(self, X, y=None, eval_set=None, drop_cols=None)

  **Description**

    Fits ``OrderedOVRClassifier`` and attaches trained models to the class pipeline.

    If self.train_final_only=True (not default), fit skips the Ordered OVR training and trains/evaluates the model using the API for OrderedOVRClassifier on all classes.

    If self.train_final_model=True (default), fit does training on remaining classes not specified in self.ovr_vals.

    Binary models are evaluated with the imported plot_thresholds function, which evaluates precision, recall, and fscores for all thresholds with 0.01 interval spacing and automatically sets the threshold at the best weighted fscore. Multiclass models are evaluated using the imported extended_classification_report function.

  **Parameters**

      X: array-like, shape = [n_samples, n_features]
          Input data for model training.

      y: array-like, shape = [n_samples, ], optional
          True labels for X. If not provided and X is a DataFrame, will extract y column from X with the provided self.target value.

      eval_set: DataFrame or list of (X, y) tuple, optional
          Dataset to use as validation set for early-stopping and/or scoring trained models.

      drop_cols: list of str, optional
          Labels of columns to ignore in modeling, only applicable to pandas DataFrame X input.

  **Returns**

      self

.. py:method:: OrderedOVRClassifier.predict(self, X, start=0, drop_cols=None)

  **Description**

    Predict multi-class targets using underlying estimators. Positive predictions from earlier steps in the prediction pipeline will be the final prediction, as this is the intended functionality of OrderedOVRClassifier.

  **Parameters**

      X: array-like, shape = [n_samples, n_features]
          Data used for predictions.

      start: int, optional, default: 0
          Index of the prediction pipeline to start on. Defaults to 0 (makes prediction through full pipeline).

      drop_cols: list of str, optional
          Labels of columns ignored in modeling, only applicable to pandas DataFrame X input.

  **Returns**

      pred: array-like, shape = [n_samples, ]
          Predicted multi-class targets.


.. py:method:: OrderedOVRClassifier.predict_proba(self, X, score_type='uniform', drop_cols=None)

  **Description**

    Predict probabilities for multi-class targets using underlying estimators. Because each classifier is trained against different classes in Ordered One-Vs-Rest modeling, it is not possible to output accurate probabilities that always return the correct prediction (from the predict function) for the most probable class. Instead, the following score_type methods are used to output probability estimates.

    If the score_type is ``'raw'``, the probability score from the specific model used to train the class of interest is returned for each class. There are no corrections applied for the 'raw' score_type and the outputted probabilities will not sum to 1.

    If the score_type is ``'chained'``, the probability of the next classifier in the pipeline is scaled down so the probabilities sum to the negative ('rest') classification probability of the current classifier.

    If the score type is ``'uniform'``, positive values for Ordered One-Vs_Rest classifications are treated in the same manner as the 'chained' score_type. Negative ('rest') outcomes always return a uniform value based on the 1-precision score for the 'rest' class of the binary model used in the pipeline step for the One-Vs-Rest classifier. This ensures that future pipeline models that sub-classify the 'rest' classification will always sum up to the same number, allowing more meaningful interpretation of the probabilities.

  **Parameters**

      X: array-like, shape = [n_samples, n_features]
          Data used for predictions.

      score_type: str, optional, default: 'uniform'
          Acceptable inputs are 'raw', 'chained', and 'uniform'.

      drop_cols: list of str, optional
          Labels of columns ignored in modeling, only applicable to pandas DataFrame X input.

  **Returns**

      pred: array-like, shape = [n_samples, n_classes]
          Returns the probability of the sample for each class in the model, where classes are ordered as they are in self._le.classes_.

Plotting API
------------

.. py:method:: OrderedOVRClassifier.plot_feature_importance(self, X, y=None, filter_class=None, n_jobs=-1, n_samples=5000, progressbar=True, drop_cols=None)

  **Description**

    Wrapper function for calling the plot_feature_importance function from skater, which estimates the feature importance of all columns based on a random sample of 5000 data points. To calculate feature importance the following procedure is executed:

    1. Calculate the original probability predictions for each class.
    2. Loop over the columns, one at a time, repeating steps 3-5 each time.
    3. Replace the entire column corresponding to the variable of interest with replacement values randomly sampled from the column of interest
    4. Use the model to predict the probabilities.
    5. The (column, average_absolute_probability_difference) becomes an (x, y) pair of the feature importance plot.
    6. Normalize the average_probability_difference so the sum equals 1.

  **Parameters**

      X: array-like, shape = [n_samples, n_features]
          Input data used for training or evaluating the fitted model.

      y: array-like, shape = [n_samples, ], optional
          True labels for X. If not provided and X is a DataFrame, will extract y column from X with the provided self.target value.

      filter_class: str or numeric, optional
          If specified, the feature importances will only be calculated for y data points matching class specified for filter_class.

      n_jobs: int, optional, default: -1
          The number of CPUs to use to compute the feature importances. -1 means 'all CPUs' (default).

      n_samples: int, optional, default: 5000
          How many samples to use when computing importance.

      progressbar: bool, optional, default: True
          Whether to display progress. This affects which function we use to multipool the function execution, where including the progress bar results in 10-20% slowdowns.

      drop_cols: list of str, optional
          Labels of columns ignored in modeling, only applicable to pandas DataFrame X input.

.. py:method:: OrderedOVRClassifier.plot_partial_dependence(self, X, col, grid_resolution=100, grid_range=(.05, 0.95), n_jobs=-1, n_samples=1000, progressbar=True, drop_cols=None)

  **Description**

    Wrapper function for calling the plot_partial_dependence function from skater, which estimates the partial dependence of a column based on a random sample of 1000 data points. To calculate partial dependencies the following procedure is executed:

    1. Pick a range of values (decided by the grid_resolution and grid_range parameters) to calculate partial dependency for.
    2. Loop over the values, one at a time, repeating steps 3-5 each time.
    3. Replace the entire column corresponding to the variable of interest with the current value that is being cycled over.
    4. Use the model to predict the probabilities.
    5. The (value, average_probability) becomes an (x, y) pair of the partial dependence plot.

  **Parameters**

      X: array-like, shape = [n_samples, n_features]
          Input data used for training or evaluating the fitted model.

      col: str
          Label for the feature to compute partial dependence for.

      grid_resolution: int, optional, default: 100
          How many unique values to include in the grid. If the percentile range is 5% to 95%, then that range will be cut into <grid_resolution> equally size bins.

      grid_range: (float, float) tuple, optional, default: (.05, 0.95)
          The percentile extrama to consider. 2 element tuple, increasing, bounded between 0 and 1.

      n_jobs: int, optional, default: -1
          The number of CPUs to use to compute the partial dependence. -1 means 'all CPUs' (default).

      n_samples: int, optional, default: 1000
          How many samples to use when computing partial dependence.

      progressbar: bool, optional, default: True
          Whether to display progress. This affects which function we use to multipool the function execution, where including the progress bar results in 10-20% slowdowns.

      drop_cols: list of str, optional
          Labels of columns ignored in modeling, only applicable to pandas DataFrame X input.

.. py:method:: OrderedOVRClassifier.plot_oovr_dependencies(self, ovr_val, X, y=None, comp_vals=None, drop_cols=None)

  **Description**

    Evaluates the effect of changing the threshold of an ordered OVR classifier against other classes with respect to accuracy, precision, recall, and f1 metrics.

  **Parameters**

      ovr_val: str, int, or float
          Class label to evaluate metrics against other classes.

      X: array-like, shape = [n_samples, n_features]
          Data used for predictions.

      y: array-like, shape = [n_samples, ], optional
          True labels for X. If not provided and X is a DataFrame, will extract y column from X with the provided self.target value.

      comp_vals: list of str, optional
          List of classes to compare against the trained classifier for ovr_val. If None, all other classes will be compared against the ovr_val class.

      drop_cols: list of str, optional
          Labels of columns ignored in modeling, only applicable to pandas DataFrame X input.

Model Selection API
-------------------

.. py:method:: OrderedOVRClassifier.fit_test(self, model, X, y=None, eval_set=None, drop_cols=None)

  **Description**

    Function for training a final model against a (possibly) classification-masked X dataset. Does not attach trained model to the pipeline for OrderedOVRClassifier. Also evaluates classification with the imported extended_classification_report function.

    Note that if an OVR model has been attached to the pipeline, the same dataset(s) used to train/evaluate the first OVR model must be used to train future OrderedOVRClassifier pipeline steps.

  **Parameters**

      model: model
          Unfitted model to test against dataset, which may have classification values masked if previous OVR training has been attached to pipeline.

      X: array-like, shape = [n_samples, n_features]
          Input data for model training.

      y: array-like, shape = [n_samples, ], optional
          True labels for X. If not provided and X is a DataFrame, will extract y column from X with the provided self.target value.

      eval_set: DataFrame or list of (X, y) tuple, optional
          Dataset to use as validation set for early-stopping and/or scoring trained models.

      drop_cols: list of str, optional
          Labels of columns to ignore in modeling, only applicable to pandas DataFrame X input.

  **Returns**

      model: OOVR_Model
          OVR fitted model trained against classification-masked X dataset.

.. py:method:: OrderedOVRClassifier.fit_test_ovr(self, model, ovr_val, X, y=None, eval_set=None, drop_cols=None)

  **Description**

    Function for training an OVR model against a (possibly) classification-masked X dataset. Does not attach trained model to the pipeline for OrderedOVRClassifier. Also evaluates binary classification with the imported plot_thresholds function, which plots precision, recall, and fscores for all thresholds with 0.01 interval spacing.

    Note that if an OVR model has been attached to the pipeline, the same dataset(s) used to train/evaluate the first OVR model must be used to train future OrderedOVRClassifier pipeline steps.

  **Parameters**

      model: model
          Unfitted model to test against dataset, which may have classification values masked if previous OVR training has been attached to pipeline.

      ovr_val: str, int, or float
          Classification value to perform OVR training.

      X: array-like, shape = [n_samples, n_features]
          Input data for model training.

      y: array-like, shape = [n_samples, ], optional
          True labels for X. If not provided and X is a DataFrame, will extract y column from X with the provided self.target value.

      eval_set: DataFrame or list of (X, y) tuple, optional
          Dataset to use as validation set for early-stopping and/or scoring trained models.

      drop_cols: list of str, optional
          Labels of columns to ignore in modeling, only applicable to pandas DataFrame X input.

  **Returns**

      model: OOVR_Model
          OVR fitted model trained against classification-masked X dataset.

.. py:method:: OrderedOVRClassifier.fit_test_grid(self, grid_model, X, y=None, eval_set=None, ovr_val=None, drop_cols=None)

  **Description**

    Wrapper for testing hyper-parameter optimization models with the OrderedOVRClassifier API against a (possibly) classification-masked X dataset.

    Note that if an OVR model has been attached to the pipeline, the same dataset(s) used to train/evaluate the first OVR model must be used to train future OrderedOVRClassifier pipeline steps.

  **Parameters**

      grid_model: GridSearchCV or RandomizedSearchCV model
          Hyper-parameter optimizer model from the sklearn.model_selection library. Must be initiated with base estimator and parameter grid.

      X: array-like, shape = [n_samples, n_features]
          Input data for model training.

      y: array-like, shape = [n_samples, ], optional
          True labels for X. If not provided and X is a DataFrame, will extract y column from X with the provided self.target value.

      eval_set: DataFrame or list of (X, y) tuple, optional
          Dataset to use as validation set for early-stopping and/or scoring trained models.

      ovr_val: str, int, or float, optional
          If specified, fit_test_grid will perform OVR modeling against the ovr_val classification label.

      drop_cols: list of str, optional
          Labels of columns to ignore in modeling, only applicable to pandas DataFrame X input.

  **Returns**

      model: OOVR_Model
          OVR fitted model trained against classification-masked X dataset.

.. py:method:: OrderedOVRClassifier.attach_model(self, oovr_model)

  **Description**

    Attaches an OVR model to the OrderedOVRClassifier prediction pipeline.

  **Parameters**

      oovr_model: OOVR_Model
          OOVR_Model object returned from fit_test of fit_test_ovr functions. OOVR_Model contains compatible OVR classifier to add to the prediction pipeline of OrderedOVRClassifier.

  **Returns**

      self

Miscellaneous API
-----------------

.. py:method:: OrderedOVRClassifier.multiclassification_report(self, X, y=None, drop_cols=None)

  **Description**

    .. _sklearn.metrics.classification_report: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html

    Wrapper function for extended_classification_report, which is an extension of sklearn.metrics.classification_report_. Builds a text report showing the main classification metrics and the total count of multiclass predictions per class.

  **Parameters**

      X: array-like, shape = [n_samples, n_features]
          Data used for predictions.

      y: array-like, shape = [n_samples, ], optional
          True labels for X. If not provided and X is a DataFrame, will extract y column from X with the provided self.target value.

      drop_cols: list of str, optional
          Labels of columns ignored in modeling, only applicable to pandas DataFrame X input.

.. py:method:: OrderedOVRClassifier.predict_json(self, row)

  **Description**

    Predict multi-class target from JSON using underlying estimators. Positive predictions from earlier steps in the prediction pipeline will be the final prediction, as this is the intended functionality of OrderedOVRClassifier.

  **Parameters**

      row: json
          Single JSON row to make prediction from.

  **Returns**

      pred: str or int
          Predicted multi-class target for input row data.

.. py:method:: OrderedOVRClassifier.predict_proba_json(self, row, score_type='uniform', print_prob=False)

  **Description**

    Predict probabilities for multi-class target from JSON using underlying estimators. Because each classifier is trained against different classes in Ordered One-Vs-Rest modeling, it is not possible to output accurate probabilities that always return the correct prediction for the most probable class. Instead, the following score_type methods are used to output probability estimates.

    If the score_type is ``'raw'``, the probability score from the specific model used to train the class of interest is returned for each class. There are no corrections applied for the 'raw' score_type and the outputted probabilities will not sum to 1.

    If the score_type is ``'chained'``, the probability of the next classifier in the pipeline is scaled down so the probabilities sum to the negative ('rest') classification probability of the current classifier.

    If the score type is ``'uniform'``, positive values for Ordered One-Vs_Rest classifications are treated in the same manner as the 'chained' score_type. Negative ('rest') outcomes always return a uniform value based on the 1-precision score for the 'rest' class of the binary model used in the pipeline step for the One-Vs-Rest classifier. This ensures that future pipeline models that sub-classify the 'rest' classification will always sum up to the same number, allowing more meaningful interpretation of the probabilities.

  **Parameters**

      row: json
          Single JSON row to make prediction from.

      score_type: str, optional, default: 'uniform'
          Acceptable inputs are 'raw', 'chained', and 'uniform'.

      print_prob: bool, optional
          Whether to print out the probabilities to console.

  **Returns**

      pred: array-like, shape = [1, n_classes] or None
          Returns the probability of the sample for each class in the model, where classes are ordered as they are in self._le.classes_ or returns None if print_prob is True.

.. py:method:: OrderedOVRClassifier.score(self, X, y=None, sample_weight=None, drop_cols=None)

  **Description**

    Returns the mean accuracy on the given test data and labels.

  **Parameters**

      X: array-like, shape = [n_samples, n_features]
          Test samples.

      y: array-like, shape = [n_samples, ], optional
          True labels for X.

      sample_weight: array-like, shape = [n_samples], optional
          Sample weights.

      drop_cols: list of str, optional
          Labels of columns ignored in modeling, only applicable to pandas DataFrame X input.

  **Returns**

      scr: float
              Mean accuracy of self.predict(X) wrt y.
