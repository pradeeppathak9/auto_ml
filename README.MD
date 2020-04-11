

```python
from estimator import Estimator
from folds import CustomFolds, FoldScheme

params={
    'n_estimators': 20000, 
    'learning_rate': 0.01,
    'boosting_type': 'gbdt', 
    'colsample_bytree': 0.80,        
    'min_child_weight': 40.0,
    'num_leaves': 16, 
    'objective': 'regression', 
    'subsample': 0.50, 
    'subsample_freq': 5,
    'metric': 'custom'
}

est = Estimator(
    LGBMRegressor(**params), early_stopping_rounds=100, random_state=100, 
    variance_penalty=1, verbose=100, eval_metric='mae', validation_scheme=FoldScheme.StratifiedKFold
)

est = lgb_model.fit_transform(X_train, y_train)
print(est.avg_cv_score)

```
