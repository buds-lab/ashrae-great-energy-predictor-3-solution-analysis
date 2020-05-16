import optuna
import numpy as np
import pandas as pd
from functools import partial
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from ashrae.utils import DATA_PATH

def load_preds(submissions_list):
    """Loads a list of submissions as an ndarray"""
    def extract_submission(file_name):
        subm = pd.read_csv(f"{DATA_PATH}/iii/submissions/{file_name}.csv")
        return subm.sort_values("row_id")[["meter_reading"]].values

    return np.concatenate([extract_submission(x) for x in submissions_list], -1)


class GeneralizedMeanBlender():
    """Combines multiple predictions using generalized mean"""

    def __init__(self, p_range=(0,1), random_state=42):
        """
        Args:
            p_range: Range of the power in the generalized mean. Defalut is (0,2).
            random_state: Seed for the random number generator.

        Returns: GeneralizedMeanBlender object
        """    
        self.p_range = p_range
        self.random_state = random_state
        self.p = None
        self.c = None
        self.weights = None
                
    def _objective(self, trial, X, y):
                    
        # create hyperparameters
        p = trial.suggest_uniform(f"p", *self.p_range)
        c = trial.suggest_uniform(f"c", 0.95, 1.05)
        weights = [
            trial.suggest_uniform(f"w{i}", 0, 1)
            for i in range(X.shape[1])
        ]

        # blend predictions
        blend_preds, total_weight = 0, 0
        if p == 0:
            for j,w in enumerate(weights):
                blend_preds += w*np.log1p(X[:,j])
                total_weight += w
            blend_preds = c*np.expm1(blend_preds/total_weight)
        else:
            for j,w in enumerate(weights):
                blend_preds += w*X[:,j]**p
                total_weight += w
            blend_preds = c*(blend_preds/total_weight)**(1/p)
            
        # calculate mean squared error
        return np.sqrt(mean_squared_error(y, blend_preds))

    def fit(self, X, y, n_trials=10): 
        # optimize objective
        obj = partial(self._objective, X=X, y=y)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(sampler=sampler)
        study.optimize(obj, n_trials=n_trials)
        # extract best weights
        if self.p is None:
            self.p = [v for k,v in study.best_params.items() if "p" in k][0]
        self.c = [v for k,v in study.best_params.items() if "c" in k][0]
        self.weights = np.array([v for k,v in study.best_params.items() if "w" in k])
        self.weights /= self.weights.sum()

    def transform(self, X): 
        assert self.weights is not None and self.p is not None,\
        "Must call fit method before transform"
        if self.p == 0:
            return self.c*np.expm1(np.dot(np.log1p(X), self.weights))
        else:
            return self.c*np.dot(X**self.p, self.weights)**(1/self.p)
    
    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)