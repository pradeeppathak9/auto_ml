import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit, train_test_split
import random
import io, pickle
from folds import CustomFolds, FoldScheme

from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRFRegressor, XGBClassifier


class_instance = lambda a, b: eval("{}(**{})".format(a, b if b is not None else {}))
        
class Estimator(object):

    def __init__(self, model, n_splits=5, random_state=100, shuffle=True, early_stopping_rounds=None, 
             validation_scheme=None, cv_group_col=None, 
             categorical_features_indices=None, variance_penalty=0, over_sampling=None,
             eval_metric=None, scoring_metric=None, verbose=100, n_jobs=-1, **kwargs
        ):
        try:
            # build model instance from tuple/list of ModelName and params
            # model should be imported before creating the instance
            self.model = class_instance(model[0], model[1])
        except Exception as e:
            # model instance is already passed
            self.model = clone(model)

        self.n_splits = n_splits
        self.random_state = random_state
        self.seed = self.random_state
        self.shuffle = shuffle
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        assert validation_scheme is not None, "Validation Scheme can't be None"
        if isinstance(validation_scheme, str):
            self.validation_scheme = FoldScheme(validation_scheme)
        else:
            self.validation_scheme = validation_scheme
        self.cv_group_col = cv_group_col
        self.categorical_features_indices=categorical_features_indices
        self.variance_penalty = variance_penalty
        self.verbose = verbose
        self.eval_metric = eval_metric
        self.scoring_metric = scoring_metric

        self.over_sampling = over_sampling


    def get_params(self):
        return {
            'model': (self.model.__class__.__name__, self.model.get_params()),
            'n_splits': self.n_splits,
            'random_state': self.random_state,
            'shuffle': self.shuffle,
            'n_jobs': self.n_jobs,
            'early_stopping_rounds': self.early_stopping_rounds,
            'variance_penalty': self.variance_penalty,
            'validation_scheme': self.validation_scheme.value if isinstance(self.validation_scheme, FoldScheme) else self.validation_scheme,
            'cv_group_col': self.cv_group_col,
            'eval_metric': self.eval_metric,
            'scoring_metric': self.scoring_metric,
            "verbose": self.verbose
        }


    def fit(self, x, y, use_oof=False, n_jobs=-1, groups=None):
        assert hasattr(self.model, 'fit'), "Model/algorithm needs to implement fit()"
        fitted_models = []
        if use_oof:
            folds = CustomFolds(num_folds=self.n_splits, validation_scheme=self.validation_scheme, shuffle=self.shuffle, 
                random_state=random.randint(0,1000) if self.random_state=='random' else self.random_state)
            self.indices = folds.split(x, y, group=self.cv_group_col)
            
            for i, (train_index, test_index) in enumerate(self.indices):
                model = clone(self.model)
                model.n_jobs = n_jobs
                x_train, y_train = x[train_index], y[train_index]
                x_val, y_val = x[test_index], y[test_index]

                if self.over_sampling:
                    x_train, y_train = self.over_sampling(x_train, y_train)

                if self.early_stopping_rounds is not None:
                    if model.__class__.__name__ in ["LGBMRegressor", "LGBMClassifier"]:
                        model.fit(X=x_train, y=y_train, eval_set=[(x_val, y_val)], eval_metric=self.eval_metric,
                            early_stopping_rounds=self.early_stopping_rounds, verbose=self.verbose)
                    elif model.__class__.__name__ in ["XGBClassifier", "XGBRegressor"]:
                        model.fit(X=x_train, y=y_train, eval_set=[(x_val, y_val)], eval_metric=self.eval_metric,
                            early_stopping_rounds=self.early_stopping_rounds, verbose=self.verbose)
                    elif model.__class__.__name__ in ["CatBoostClassifier"]:
                        model.od_wait=int(self.early_stopping_rounds)
                        model.fit(x_train, y_train, cat_features=self.categorical_features_indices,
                             eval_set=(x_val, y_val ), use_best_model=True, verbose=self.verbose)
                    else:
                        raise ValueError("Early stopping not implemented for the given model!")
                else:
                    model.fit(x_train, y_train)
                        
                fitted_models.append(model)
        else:
            model = clone(self.model)
            model.n_jobs = n_jobs
            
            if self.early_stopping_rounds is not None:
                x_train, x_val, y_train, y_val = train_test_split(x, y, test_size =0.2, shuffle=True,
                    random_state=random.randint(0,1000) if self.random_state=='random' else self.random_state
                )
                if model.__class__.__name__ in ["LGBMRegressor", "LGBMClassifier"]:
                    model.fit(X=x_train, y=y_train, eval_set=[(x_val, y_val)], eval_metric=self.eval_metric,
                        early_stopping_rounds=self.early_stopping_rounds, verbose=self.verbose)
                elif model.__class__.__name__ in ["XGBClassifier", "XGBRegressor"]:
                    model.fit(X=x_train, y=y_train, eval_set=[(x_val, y_val)], eval_metric=self.eval_metric,
                        early_stopping_rounds=self.early_stopping_rounds, verbose=self.verbose)
                elif model.__class__.__name__ in ["CatBoostClassifier"]:
                    model.od_wait=int(self.early_stopping_rounds)
                    model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True,
                        cat_features=self.categorical_features_indices,  verbose=self.verbose)
                else:
                    raise ValueError("Early stopping not implemented for the given model!")
            else:
                model.fit(x, y)
            fitted_models.append(model)
            
        self.fitted_models = fitted_models
        return self
    

    def feature_importances(self):
        assert hasattr(self, 'fitted_models'), "Model/algorithm needs to implement fit()"
        if self.model.__class__.__name__ == "LogisticRegression":
            feature_importances = np.column_stack(m.coef_[0] for m in self.fitted_models)
        if self.model.__class__.__name__ == "LinearRegression":
            feature_importances = np.column_stack(m.coef_ for m in self.fitted_models)
        else:
            feature_importances = np.column_stack(m.feature_importances_ for m in self.fitted_models)
        importances = np.mean(1.*feature_importances/feature_importances.sum(axis=0), axis=1)
        return importances

    def feature_importance_df(self, columns=None):
        importances = self.feature_importances()
        if columns is not None:
            assert len(columns) != len(importances), "Columns length Mismatch!"
            df = pd.DataFrame(list(zip(columns, importances)), columns=['column', 'feature_importance'])
        else:
            df = pd.DataFrame(list(zip(list(range(len(importances))), importances)), columns=['column_index', 'feature_importance'])
        df.sort_values(by='feature_importance', ascending=False, inplace=True)
        df['rank'] = np.arange(len(importances)) + 1
        return df

    def transform(self, x):
        assert hasattr(self, 'fitted_models'), "Model/algorithm needs to implement fit()"
        return np.mean(np.column_stack((
            est.predict_proba(x)[:,1] if hasattr(est, "predict_proba") else est.predict(x) for est in self.fitted_models 
        )), axis=1)

    def fit_transform(self, x, y, groups=None):
        self.fit(x, y, use_oof=True)
        predictions = np.zeros((x.shape[0],))
        for i, (train_index, test_index) in enumerate(self.indices):
            if hasattr(self.fitted_models[i], "predict_proba"):
                predictions[test_index] = self.fitted_models[i].predict_proba(x[test_index])[:,1]
            else:
                predictions[test_index] = self.fitted_models[i].predict(x[test_index])

        self.cv_scores = [
            self.scoring_metric(y[test_index], predictions[test_index])
            for i, (train_index, test_index) in enumerate(self.indices)
        ]
        self.avg_cv_score = np.mean(self.cv_scores)
        return predictions

    def is_regression(self):
        return isinstance(self.model, LogisticRegression)

    def save_model(self, file_name=None):
        """ Saving fitted model and Estimator params for reuse!"""
        assert self.fitted_models and len(self.fitted_models) > 0, "Cannot save a model that is not fitted"
        assert file_name, "file_name cannot be None"
        with open(file_name, "wb") as out_file:
            pickle.dump({"fitted_models": self.fitted_models, "params": self.get_params()}, out_file)
            return file_name

    @staticmethod
    def load_model(file_name=None):
        """ Loads a model from saved picke of fitted models and Estimator params. returns an Estimator!"""
        assert file_name, "file_name cannot be None"
        _dict = pickle.load(open(file_name, "rb"))
        est_ = Estimator(**_dict['params'])
        est_.fitted_models = _dict['fitted_models']
        return est_

    def predict_proba(self, x):
        return self.transform(x)

    def get_repeated_out_of_folds(self, x, y, num_repeats=1):
        cv_scores, fitted_models, indices = [], [], []
        for iteration in range(num_repeats):
            self.random_state = random.randint(0, 1000) if self.random_state=='random' else self.seed*(iteration+1)
            predictions = self.fit_transform(x, y)
            cv_scores+=self.cv_scores
            self.random_state = 'random' if self.random_state=='random' else self.seed
            fitted_models.extend(self.fitted_models)
            indices.extend(self.indices)
        self.fitted_models = fitted_models
        self.indices = indices
        return {
            'cv_scores': cv_scores,
            'avg_cv_score': np.mean(cv_scores),
            'var_scores': np.std(cv_scores),
            'eval_score': np.mean(cv_scores) - (self.variance_penalty*np.std(cv_scores))
        }

    def get_nested_scores(self, x, y):
        pass
