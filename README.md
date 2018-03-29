# OrderedOVRClassifier
API for performing Ordered One-Vs-Rest Classification with scikit-learn

## 📖 Documentation

<table>
<tr>
    <td>
    	<a href=http://nbviewer.jupyter.org/github/alvinthai/OrderedOVRClassifier/blob/master/examples/example.ipynb>Tutorial</a>
    </td>
    <td>How to use OrderedOVRClassifier</td>
</tr>
<tr>
    <td>
    	<a href=https://alvinthai.github.io/OrderedOVRClassifier/api_reference.html>API Reference</a>
    </td>
    <td>The detailed reference for OrderedOVRClassifier's API</td>
</tr>
</table>


## Introduction

OrderedOVRClassifier is a custom scikit-learn module for approaching multi-classification with an Ordered One-Vs-Rest modeling approach. Ordered One-Vs-Rest Classification performs a series of One-Vs-Rest Classifications where negative results are moved into subsequent training with previous classifications filtered out.

Most muliti-classification machine learning algorithms use an [One-Vs-The-Rest](http://scikit-learn.org/stable/modules/multiclass.html#one-vs-the-rest) strategy to decompose multiclass problems into ``n_class`` binary classification problems. For a dataset with ``n_class = 5``, this means that the machine learning algorithm is training models as such:

```
Model 1:      Class [1] vs Classes [2, 3, 4, 5]
Model 2:      Class [2] vs Classes [1, 3, 4, 5]
Model 3:      Class [3] vs Classes [1, 2, 4, 5]
Model 4:      Class [4] vs Classes [1, 2, 3, 5]
Model 5:      Class [5] vs Classes [1, 2, 3, 4]
```

OrderedOVRClassifier provides an alternate paradigm for approaching multi-classification. For a dataset with ``n_class = 5``, the training steps could look like:

```
Model 1:      Class [1] vs Classes [2, 3, 4, 5]
Model 2:      Class [2] vs Classes [3, 4, 5]
Model 3/4/5:  Class [3] vs Classes [4, 5]
              Class [4] vs Classes [3, 5]
              Class [5] vs Classes [3, 4]
```

Why would you want to model a multi-classification problem with a Ordered One-Vs-Rest approach? There are several use cases:
- Perhaps some classes have high predictive accuracy and others not so much. If we have a binary model that screens out the highly predictive class, we can speed up the training for the remaining classes by reducing the number of classification models for training steps that require heavy optimization.

- Maybe we are willing to sacrifice the precision or recall performance of the predictions from one class to improve the precision or recall in another. Changing the threshold for binary classification allows us to do this.

- It could be that different algorithms perform better for specific classes. Ordered One-Vs-Rest classification does not require the same machine learning algorithm to be used for all classes, giving us the flexibility to mix different algorithms for classifying different classes.

With Ordered One-Vs-Rest classification, positive predictions from earlier modeling steps always take precedence in the final predictions.


## Features

The API for OrderedOVRClassifier is designed to be user-friendly with pandas, numpy, and scikit-learn. There is also built in functionality to support easy handling for early stopping on the sklearn wrapper for XGBoost and LightGBM.

OrderedOVRClassifier could also be used to train multi-classification problems without an Ordered One-Vs-Rest strategy by setting ``train_final_only=True``, allowing the user to take advantage of the general and convenience features listed below for their own modeling purposes.

- Ordered One-Vs-Rest Classification Features
  - Reduce training time by training models on fewer classes.
  - Tradeoff accuracy/precision/recall between different classes.
  - Mix classification algorithms for different classes.
- General Features
  - Model-agnostic calculation of feature importances.
  - Model-agnostic calculation of partial dependence.
  - Instantly evaluate the precision/recall/f1 scores for each class when making predictions.
- Convenience Features
  - Train and evaluate results from pandas DataFrames without specifying y input.
  - Simple interface for passing in evaluation datasets for early stopping in LightGBM and XGBoost.
  - Attach models stepwise instead of training the full model in the fit step.


## Quickstart

To use OrderedOVRClassifier for Ordered One-Vs-Rest Classification, the ordered steps for the classification must be specified with the ``ovr_vals`` parameter and the model(s) used to train the binary classifiers must be specified in the ``model_dict`` parameter. The model used to train the remaining classes should be specified with a ``'final'`` key in the ``model_dict``. Refer to the [tutorial](http://nbviewer.jupyter.org/github/alvinthai/OrderedOVRClassifier/blob/master/examples/example.ipynb) for more specific usage examples.

```python
from OrderedOVRClassifier import OrderedOVRClassifier

ovr_vals = ['1st class', '2nd class']

model_dict = {'1st class': RandomForestClassifier(),
              '2nd class': RandomForestClassifier(),
              'final': XGBClassifier()}

oovr = OrderedOVRClassifier(target='output', ovr_vals=ovr_vals, model_dict=model_dict)
```

#### Fitting Models to Our Data

If working with pandas DataFrames, this is as simple as passing in the training dataset into the ``X`` parameter. An optional test dataset could be passed in similarly into the ``eval_set`` parameter. When working with numpy arrays, passing in ``X`` and ``y`` as usual is required.

```python
oovr.fit(train_df, eval_set=test_df)
```

After fitting, we have lots of nice methods and properties attached to the fitter object.

## Visualization and Model Evaluation

OrderedOVRClassifier has a simple interface for plotting feature importances and partial dependencies. Precision and recall can also easily be evaluated and plotted against the threshold for binary classification of another class. Refer to the [tutorial](http://nbviewer.jupyter.org/github/alvinthai/OrderedOVRClassifier/blob/master/examples/example.ipynb#Plot-Feature-Importance) to see example visualization outputs.

```python
# plot model-agnostic feature importances
oovr.plot_feature_importance(train_df)

# plot model-agnostic partial dependence with respect to one column
oovr.plot_partial_dependence(train_df, 'some_column')

# generate multi-classification precision/recall/f1/accuracy report
oovr.multiclassification_report(test_df)

# plot threshold dependent accuracy/precision/recall/f1
oovr.plot_oovr_dependencies('some_class', test_df)
```

## Using OrderedOVRClassifier Modularly

OrderedOVRClassifier can be fit without training the full model pipeline. We could omit running the ``fit`` step altogether or fit an incomplete pipeline and instead use the ``fit_test`` or ``fit_test_ovr`` methods to train candidate models for attachment.

```python
best_lgb = LGBMClassifier(n_estimators=100, num_leaves=250, min_child_samples=5,
                          colsample_bytree=1.0, subsample=0.8)

final_model = oovr.fit_test(best_lgb, train_df, eval_set=test_df)
```

The objects returned from the ``fit_test`` or ``fit_test_ovr`` methods can be attached to the OrderedOVRClassifier object with the ``attach_model`` method.
```python
oovr.attach_model(final_model)  
```

Refer to the [API Reference](https://alvinthai.github.io/OrderedOVRClassifier/api_reference.html#model-selection-api) and the [tutorial](http://nbviewer.jupyter.org/github/alvinthai/OrderedOVRClassifier/blob/master/examples/example.ipynb#Test-and-Attach-Models) for more details.

## Dependencies

OrderedOVRClassifier is tested on Python 2.7.13 and depends on numpy (≥1.13.3),
pandas (≥0.21.1), scikit-learn (≥0.19.1), matplotlib (≥2.1.1), and skater(≥1.0.3). I have not tested the codebase against earlier versions of these packages.


## 💬 Feedback / Questions

<table>
<tr>
	<td><b>Feature Requests / Issues</b></td>
    <td>
    	<a href=https://github.com/alvinthai/OrderedOVRClassifier/issues>https://github.com/alvinthai/OrderedOVRClassifier/issues</a>
    </td>
</tr>
<tr>
	<td><b>Email</b></td>
    <td>alvinthai@gmail.com</td>
</tr>
</table>
