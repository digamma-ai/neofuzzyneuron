import time
import numpy as np

from .mfunc import *


class NeoFuzzyNeuron:  
    """
    Neo-fuzzy neuron model

    Supports uniform and quantile grids, linear membership functions and
    full batch RPROP training with early stopping.

    Attributes
        n_rules (int): number of fuzzy inference rules in the neuron
        uniform (bool): if True, the uniform grid of fuzzy partitions is used;
            if False, the input domain is partitioned by quantiles of the 
            unconditional time series distribution. Uniform grid is makes evaluation
            and training ~4.0x faster.
        n_epochs (int): number of training iteration
        max_no_best (int): early stopping parameter        
        write_log (bool): if True, training process (train/test MSE, time per epoch)
            is written into a ``log_`` field
    """
    
    def __init__(self, n_rules, uniform = True, 
                 n_epochs = 1000, max_no_best = 16, write_log = True):   
        self.n_rules  = n_rules       
        self.n_epochs = n_epochs
        self.max_no_best = max_no_best
        self.uniform = uniform
        
        self._weights = None
        self._write_log = write_log       
        self._mse = 0.0        
        
        self.log_ = list()


    def _data_initializer(self, X, y, grid):
        _, n_inputs = X.shape
        a = grid[ :-2,:]
        b = grid[2:  ,:]
        mask = np.logical_and(
            X[:, None, :] >= a[None, :, :],
            X[:, None, :] <  b[None, :, :]
        )
        W = np.sum(y[:, None, None] * mask, axis = 0) 
        W /= n_inputs * np.maximum(np.sum(mask, axis = 0), 1)
        return W
        
        
    def _initialize(self, X, y):
        """
        Special initialization method for regression tasks. 

        Initializes weights by averages over crisp supports of 
        corresponding fuzzy partitions. Also sets initial RPROP steps
        to 1e-4 to ensure smooth convergence from the initial point,
        as this initialization method has proven to be quite good.

        Args
            X (ndarray): input data
            y (ndarray): target variable
        """

        grid = (uniform_grid if self.uniform else density_grid)(X, self.n_rules)
        self._mfunc = (get_sym_mfunc if self.uniform else get_asym_mfunc)(grid)
        self._weights = self._data_initializer(X, y, grid)
        self._weights_update = np.zeros_like(self._weights)
        self._step = 1e-4 * np.ones_like(self._weights)
        self._gradients = np.ones_like(self._weights, dtype = int)


    def _predict_batch(self, X, memberships = None):
        if memberships is None:
            memberships = self._mfunc(X)
        weights = self._weights[None,:,:]
        y = np.sum(weights * memberships, axis = (1,2))
        return y


    def _gradient(self, errors, M):
        E = errors[:, None, None]
        G = E * M    
        return np.mean(G, axis = 0) 
    
    
    def _rprop_iter(self, X_train, y_train, inc = 1.2, dec = 0.5): 
        memberships = self._mfunc(X_train)
        y_pred = self._predict_batch(X_train, memberships)
        errors = y_train - y_pred
        mse = np.mean(np.square(errors))
        
        gradients = np.sign(self._gradient(errors, memberships))
        
        sign = gradients * self._gradients
        no_sign_change = sign >= 0
        sign_change = sign < 0
        
        self._step[sign > 0] *= inc
        self._step[sign_change] *= dec
        
        weights_update = self._weights_update.copy()
        weights_update[no_sign_change] = -self._step[no_sign_change] * gradients[no_sign_change]
        
        self._weights[no_sign_change] -= weights_update[no_sign_change]       
        if mse > self._mse:
            self._weights[sign_change] += self._weights_update[sign_change]
            
        self._weights_update = weights_update.copy() 
        self._gradients = gradients.copy()
        self._mse = mse
        return mse
                   
    
    def fit(self, X_train, X_test, y_train, y_test): 
        self._initialize(X_train, y_train)
        
        best = (-1, np.inf, self._weights)
        n_no_best = 0 
        i = 0    

        while i < self.n_epochs or self.n_epochs < 0:
            t = time.time()
            
            self._rprop_iter(X_train, y_train)            
            y_test_pred = self._predict_batch(X_test)
            mse_test = np.mean(np.square(y_test - y_test_pred))

            if self.max_no_best > 0:
                if mse_test < best[1] - 1e-8:
                    best = (i, mse_test, self._weights.copy())
                    n_no_best = 0
                else:
                    n_no_best += 1     
                    
                if n_no_best > self.max_no_best:
                    self._weights = best[2].copy()
                    t = time.time() - t
                    if self._write_log:
                        self.log_.append((i, self._mse, mse_test, t))
                    break
                    
            t = time.time() - t  
            if self._write_log:
                self.log_.append((i, self._mse, mse_test, t))
            i += 1

        return self

        
    def predict(self, X): 
        memberships = self._mfunc(X)
        weights = self._weights[None,:,:]
        return np.sum(weights * memberships, axis = (1,2))


    def fit_predict(self, ts):
        return self.fit(ts).predict(ts)
