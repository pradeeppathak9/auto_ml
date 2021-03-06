### Usage:

##### Imports
```python
from estimator import Estimator
from folds import CustomFolds, FoldScheme
from hyperopt_model_selection import HyperOptModelSelection, hp, lgbm_space
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from auto_ml import AutoML
```

##### Estimator Object
```python
est = Estimator(
    LGBMRegressor(), early_stopping_rounds=100, random_state=100, 
    variance_penalty=1, verbose=100, eval_metric='mae', validation_scheme=FoldScheme.StratifiedKFold
)

est = lgb_model.fit_transform(X_train, y_train)
print(est.avg_cv_score)
```


##### Hyperparameter Tuning (HyperOpt)
```python
est = Estimator(
    LGBMClassifier(), early_stopping_rounds=100, random_state=100, 
    variance_penalty=1, verbose=100, eval_metric='roc', validation_scheme=FoldScheme.StratifiedKFold, n_splits=5
)
hpt = HyperOptModelSelection(model=est, space=lgbm_space, max_evals=50, is_maximize=True)
hpt.fit(X_train, y_train)
print(hpt.best_score)
print(hpt.best_params)
est = hpt.best_estimator

```


##### AutoML
```python
aml = AutoML(config_file_path="config.yml", validation_scheme=FoldScheme.StratifiedKFold, n_splits=5)
aml.fit(X_train, y_train)
# predictions on Test Data
aml.predict(X_test)

```


## Development in progress ....
