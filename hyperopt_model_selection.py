from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
import datetime
import random
import pickle
import numpy as np
import pandas as pd
from estimator import Estimator
from folds import FoldScheme
import copy
import random


def evaluate(X, y, model_estimator, num_repeats=1, fs_individual=None):
    '''
    evaluation function for genetic feature creation and selection
    X = train feature dataset of type np.ndarray
    y = train target array
    model_estimator = instance of Estimator class
    num_repeats = num_repeats params for get_repeated_out_of_folds method
    fs_individual = feature selection individual, list of column indicies
    ''' 
    # handle feature selection individual here....
    if fs_individual is not None:
        # individual is list of columns to subset
        X_temp = X_temp[:, fs_individual]
    return model_estimator.get_repeated_out_of_folds(X, y, num_repeats=num_repeats)



import logging, sys
logger = logging.getLogger("hyperopt_model_selection")

def setup_logging(level=logging.DEBUG, log_file=None):
    logger.setLevel(level)
    logger.handlers = []
    # adding a console handler for printing out logs
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    # aadding file handler
    if log_file is not None:
        file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=(1048576*5), backupCount=7)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

'''
follows the sample space for different to use.
'''

'''
use catb_default_space, default parameter space for catboost because catb works well eve without tuning
for tuning, use catb_exhaustive_space
'''
catb_default_space={
            'depth':hp.choice('depth',[6]),
            'one_hot_max_size':hp.choice('one_hot_max_size',[2]),
            'l2_leaf_reg': hp.choice('l2_leaf_reg', [3]),
            'iterations':hp.choice('iterations',[5000]),
            'od_type':hp.choice('od_type',['Iter']),
            'od_wait':hp.choice('od_wait',[100]),
            'eval_metric':hp.choice('eval_metric',['AUC'])
                   }

catb_exhaustive_space ={
            'depth':hp.quniform('depth',1,8,1),
            'one_hot_max_size':hp.quniform('one_hot_max_size',2,5,1),
            'l2_leaf_reg': hp.quniform('l2_leaf_reg', 3, 27, 3),
            'iterations':hp.choice('iterations',[5000]),
            'od_type':hp.choice('od_type',['Iter']),
            'od_wait':hp.choice('od_wait',[100]),
            'eval_metric':hp.choice('eval_metric',['AUC'])
                       }

lr_space={
    'penalty':hp.choice('penalty',['l1','l2']),
    'C':hp.loguniform('C',-3*np.log(10),4*np.log(10))
}

rf_space = {
    'max_depth':  hp.quniform('max_depth', 2, 10, 1),
    'max_features': hp.quniform('max_features',.2, .8, .1),
    'n_estimators': hp.quniform('n_estimators', 25, 1000, 1),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'min_samples_leaf': hp.quniform('min_samples_split', 2, 20, 1),
    'n_jobs': -1
}


xgb_space = {
#     'learning_rate': 0.01,
    # 'n_estimators': hp.quniform('n_estimators', 25,500, 1),
    # 'early_stopping_rounds':50,

#     'n_estimators': 5000,
    'max_depth':  hp.quniform('max_depth', 2, 5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 20, 2),
    'subsample': hp.quniform('subsample', 0.5, 1, .1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.1),
    'gamma': hp.quniform('gamma', 0, 1, 0.1),
#     'objective': 'binary:logistic',
    # 'n_jobs': -1
}



lgbm_space = {
#    'learning_rate': hp.quniform('learning_rate', 0.025, 0.1, 0.025),
    # 'n_estimators': hp.quniform('n_estimators', 25, 500, 1),
    # 'early_stopping_rounds': 50,
  
#     'n_estimators': 5000,
    'num_leaves':  hp.quniform('num_leaves', 16, 96, 16),
    'min_child_weight': hp.quniform('min_child_weight', 1, 20, 2),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.1),
    'subsample_freq': 5,
    
#     'objective': 'binary',
#     'boosting_type': 'gbdt',
#     'learning_rate': 0.01,
    'n_jobs': -1
}


lr_space={
    'penalty': hp.choice('penalty',['l1']),
    'C':hp.loguniform('C',-3*np.log(10),4*np.log(10)),
    # 'n_jobs': -1
}


'''
Added a class variable best_model to store the model with the best params without fitting on the dataset
'''
class HyperOptModelSelection(object):
    '''
    HyperOpt Model Selection Class for a give Model Instance and Space
    '''
    def __init__(self, model, space, max_evals=200, random_seed=100, is_maximize=True,
        warm_start=False, trials_file_path=None, num_repeats=1, log_file_path=None,
        params_mapping = {'n_estimators':int, 'num_leaves':int, 'max_depth':int, 'min_samples_leaf':int},
        ):

        self.model_estimator  = Estimator(model) if not model.__class__.__name__ == "Estimator" else model
        '''
        model: model for which the hyperparamter tuning is to be done
        space: space for the model
        max_evals: number of rounds to run (put max_evals as 1 if you're passing catb_default_space for hyperparameter tuning)
        trials_file_path: file to dump trails of the experiment, also loads if present to continue from where it left
        log_file_path: file path for dumping logs
        params_mapping: dict of params_key and a callable function which to map the value of the key
        '''
        if isinstance(space, dict):
            self.space = space            
        elif isinstance(space, str):
            self.space = eval(space)
        else:
            raise ValueError("space not defined!")
        self.max_evals = max_evals
        self.trials_file_path = trials_file_path
        self.trials = self.load_trails()
        self.log_file_path = log_file_path
        self.params_mapping = params_mapping
        self.random_seed=random_seed
        self.warm_start = warm_start
        self.num_repeats = num_repeats
        self.is_maximize = is_maximize
        self.columns = None

        setup_logging(log_file=log_file_path)


    def _params_mapping(self, params):
        for key, mapping_func in list(self.params_mapping.items()):
            if key in params: params[key] = mapping_func(params[key])
        return params
    '''
    objective function to calculate the score for every parameter combination
    To incorporate unstable models with high variance, a penalization term is added
    '''
    def objective(self, params):
        params = self._params_mapping(params)
        self.iteration+=1
        logger.debug("\nIteration: {}, Training with params: {}".format(self.iteration, params))
        model_estimator_params = self.model_estimator.get_params()
        # updating model params
        model_estimator_params['model'][1].update(params)
        model_estimator = Estimator(**model_estimator_params)
        score = evaluate(self.x, self.y, model_estimator, num_repeats=self.num_repeats, fs_individual=self.columns)
        
        loss = -1*score['eval_score'] if self.is_maximize else score['eval_score']

        logger.debug("Score - {}, Std - {}, Eval Score - {}".format(score["avg_cv_score"], np.std(score['cv_scores']), score["eval_score"]))
        logger.debug("Score across folds - {}.".format(score["cv_scores"]))
        return { 'loss': loss, 'status': STATUS_OK, 'misc': score}

    def __call__(self):
        self.iteration=0
        best = fmin(self.objective, self.space, algo=tpe.suggest, trials=self.trials, max_evals=self.max_evals, rstate=np.random.RandomState(self.random_seed))
        return best

    def fit(self, X, y, columns=None):
        logger.info("Starting HyperOpt {} Evals with Dataset of Shape ({},{})".format(self.max_evals, X.shape, y.shape))
        self.x, self.y = copy.deepcopy(X), copy.deepcopy(y)
        self.columns = columns
        if not self.warm_start: self.trials = Trials()

        best = self()
        self.save_trails()

        self.best_score = self.get_best_result()['misc']['eval_score']
        self.best_params = self.get_best_params()
        logger.info("Best Score- {}, Best Params- {}".format(self.best_score, self.best_params))

        model_estimator_params = self.model_estimator.get_params()
        model_estimator_params['model'][1].update(self.best_params)
        self.best_estimator = Estimator(**model_estimator_params)
        self.best_model = self.best_estimator.model
        del self.x, self.y
        return self

    def get_params(self, t_id):
        return self._params_mapping({k:v for k,v in list(space_eval(self.space, {k:v[0] for k, v in list(self.trials.trials[t_id]['misc']['vals'].items())}).items())})

    def get_model(self, t_id):
        return self.model(**self._params_mapping({k:v for k,v in list(space_eval(self.space, {k:v[0] for k, v in list(self.trials.trials[t_id]['misc']['vals'].items())}).items())}))

    def get_best_params(self):
        return self._params_mapping({k:v for k,v in list(space_eval(self.space, self.trials.argmin).items())})

    def get_best_result(self):
        return self.trials.best_trial['result']

    def get_performance_df(self):
        trials = []
        for t in self.trials:
            _dict = {'tid': t['tid']}
            t_id = t['tid']
            _dict.update(self._params_mapping({k:v for k,v in list(space_eval(self.space, {k:v[0] for k, v in list(self.trials.trials[t_id]['misc']['vals'].items())}).items())}))
            _dict.update(t['result']['misc'])
            del _dict['cv_scores']
            trials.append(_dict)
        return pd.DataFrame(trials)

    def save_trails(self):
        if self.trials_file_path is not None:
            pickle.dump(self.trials, open(self.trials_file_path, "wb"))

    def load_trails(self):
        try:
            if self.trials_file_path is not None:
                return pickle.load(open(self.trials_file_path, "rb"))
        except:
            pass
        return Trials()
