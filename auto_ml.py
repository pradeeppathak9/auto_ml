import yaml 
import pandas as pd
from estimator import Estimator
from hyperopt_model_selection import HyperOptModelSelection, hp

class AutoML: 
    
    def __init__(self, **kwargs):
        if kwargs.get('config'):
            self.config = kwargs.get('config')
        elif kwargs.get('config_file_path'):
            # reading model configurations from file
            self.config = yaml.load(open(kwargs.get('config_file_path')))
        else:
            raise ValueError("Requires config or config_file_path")
            
        self.validation_scheme = kwargs.get('validation_scheme')
        self.n_splits = kwargs.get('n_splits')
        self.fitted_objects = None 
            
    def fit(self, X_train, y_train):
        config = self.config.copy()
        self.fitted_objects = {}
        for model in config["Pipelines"]["Model Layer"]:
            name = model['name']
            if model["source"] == "Estimator":
                # creating estimator instance
                est = Estimator(model["model"], **model["params"], 
                                validation_scheme=self.validation_scheme, n_splits=self.n_splits)

                # performing Hyperparameter tuning if present
                if model.get('hpt'):
                    hpt = HyperOptModelSelection(model=est, **model['hpt']['params'])
                    hpt.fit(X_train, y_train)
                    est = hpt.best_estimator

                # fitting estimator
                est.fit_transform(X_train, y_train)
                
                # saving serialized fitted model
                self.fitted_objects[name] = est.to_serialized_object()
                
                if model.get("pickle_path"):
                    # saving model pickle, used for infrencing
                    est.save_model(model.get("pickle_path"))
    
    def predict(self, X_test):
        config = self.config.copy()
        df = pd.DataFrame({"id": range(len(X_test))})
        for model in config["Pipelines"]["Model Layer"]:
            name = model['name']
            if model["source"] == "Estimator":
                # creating estimator instance
                if hasattr(self, 'fitted_objects'):
                    # loading from serialized object in memory
                    est = Estimator.from_serialized_object(self.fitted_objects[name])
                elif model.get("pickle_path"):
                    # loading model from saved pickle
                    est = Estimator.load_model(model.get("pickle_path"))
                else: 
                    raise ValueError("Model not fitted/File Not found!")
                                    
                # generating predictions
                df[name] = est.transform(X_test)
                
        return df
                
