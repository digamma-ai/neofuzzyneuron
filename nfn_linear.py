import numpy as np

from sklearn.linear_model import Ridge
from copy import deepcopy
from .mfunc import uniform_grid, density_grid, get_sym_mfunc, get_asym_mfunc


def _check_solver(solver):
    try:
        solver.coef_
        solver.fit
        solver.predict
    except:
        raise ValueError("Solver does not implement `fit` or `predict` method, or `coef_` field")


class NeoFuzzyNeuron:  
    """
    Neo-fuzzy neuron model
    """    
    
    def __init__(self, n_rules, uniform = True, solver = None):   
        self.n_rules  = n_rules       
        self.uniform = uniform
        self.weights_ = None
        
        if solver is not None:
            _check_solver(solver)
            self.solver_ = deepcopy(solver)
        else:
            self.solver_ = Ridge(alpha = 1e-4)      
                   
    
    def fit(self, X, y): 
        if self.uniform:
            grid = uniform_grid(X, self.n_rules)
            self._mfunc = get_sym_mfunc(grid)
        else:
            grid = density_grid(X, self.n_rules)
            self._mfunc = get_asym_mfunc(grid)
        
        M = self._mfunc(X)
        M = M.reshape(M.shape[0], M.shape[1] * M.shape[2])
        
        self.solver_.fit(M, y)
        self.weights_ = self.solver_.coef_.copy()
        self.weights_ = self.weights_.reshape((self.n_rules + 1, X.shape[1]))
        self.c_ = self.solver_.intercept_
        return self

        
    def predict(self, X): 
        M = self._mfunc(X)
        weights = self.weights_[None,:,:]
        return np.sum(weights * M, axis = (1,2)) + self.c_


    def fit_predict(self, ts):
        return self.fit(ts).predict(ts) 