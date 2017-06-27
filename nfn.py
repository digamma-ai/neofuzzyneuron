import numpy as np

from sklearn.linear_model import Ridge
from copy import deepcopy


def rollwin(x, window, stride = 1):
    """
    Transforms a 1D-array into rolling window samples

    Uses fast numpy shape manipulations and performs in O(1) time.

    Args
        x (ndarray): input array
        window (int): window size
        stride (int): window step

    Returns
        ndarray: 2D-array of shape ``(x.size - window, window)``, 
            where rows as consequitive samples from the array ``x``

    Raises
        ValueError: if input array is shorted than ``window``
    """

    if x.size < window:
        raise ValueError("Array must containt at least %i values" % window)
    dsize   = x.dtype.itemsize
    strides = (stride*dsize, dsize)
    shape   = ((x.size - window)//stride + 1, window)
    return np.lib.stride_tricks.as_strided(x, strides = strides, shape = shape)


def uniform_grid(X, n_rules):
    """
    Generates a uniform grid from a set of input variables

    Args
        X (ndarray): row-based data matrix
        n_rules (int): number of fuzzy inference rules;
            the resulting grid has ``n_rules + 3`` points

    Returns
        ndarray: 2D-array of grid points per input variable

    Raises
        ValueError: if ``n_rules < 1``
    """

    if n_rules < 1:
        raise ValueError("Number of inference rules must be > 0")
    
    x_min = X.min(axis = 0)
    x_max = X.max(axis = 0)     
    step = (x_max - x_min) / n_rules
   
    grid = np.vstack(np.linspace(a - h, b + h, n_rules + 2) 
                     for a, b, h in zip(x_min, x_max, step))
    return grid.T


def density_grid(X, n_rules): 
    """
    Generates a density-based grid from a set of input variables

    Grid per varible is build as a set of ``n_rules + 3`` quantiles

    Args
        X (ndarray): row-based data matrix
        n_rules (int): number of fuzzy inference rules

    Returns
        ndarray: 2D-array of grid points per input variable

    Raises
        ValueError: if ``n_rules < 1``
    """

    if n_rules < 1:
        raise ValueError("Number of inference rules must be > 0")

    grid = list()
    for i in range(X.shape[1]):
        quantiles = np.percentile(X[:,i], q = np.linspace(0, 100, n_rules))
        pad_left  = 2*quantiles[ 0] - quantiles[ 1]
        pad_right = 2*quantiles[-1] - quantiles[-2]        
        grid.append(np.hstack([pad_left, quantiles, pad_right]))
    grid = np.vstack(grid).T
    return grid  


def get_sym_mfunc(grid):
    """
    Creates symmetric triangular membership functions
    for a uniform grid

    Resulting function accepts batches of data. 
    Faster than asymmetric functions

    Args
        grid (ndarray): uniform fuzzy partioning grid 

    Returns
        function: a batch processing function
    """

    _grid = grid.copy()

    a = _grid[None,  :-2, :]
    b = _grid[None, 1:-1, :]
    
    p = -1/(b - a)
    q = b
    
    def _mfunc(batch):          
        X = batch[:,None,:]
        M = np.maximum(0, p*np.abs(X - q) + 1)
        return M
    
    return _mfunc


def get_asym_mfunc(grid):
    """
    Creates asymmetric triangular membership functions
    for a density-based grid

    Resulting function accepts batches of data. 

    Args
        grid (ndarray): non-uniform fuzzy partioning grid 

    Returns
        function: a batch processing function
    """

    _grid = grid.copy()

    a = _grid[None,  :-2, :]
    b = _grid[None, 1:-1, :]
    c = _grid[None, 2:  , :]
    
    p =  1 / (b - a)
    q = -a / (b - a)
    r = -1 / (c - b)
    s =  c / (c - b)
    
    def _mfunc(batch):
        X = batch[:,None,:]
        lmask = np.ravel((X >= a) & (X < b))
        rmask = np.ravel((X >= b) & (X < c))
        M = np.zeros(rmask.size)
        M[lmask] = np.ravel(p*X + q)[lmask]
        M[rmask] = np.ravel(r*X + s)[rmask]
        M = M.reshape((X.shape[0], a.shape[1], X.shape[2]))
        return M
    
    return _mfunc


def data_initializer(X, y, grid):
    """
    Returns NFN weights initialized by averages over each fuzzy partition

    Args: 
        X (2d ndarray): Input sample
        y (1d ndarray): target values for X
        grid (2d ndarray): fuzzy partitioning grid

    Returns:
        2d ndarray: initialized NFN weights
    """

    _, n_inputs = X.shape
    a = grid[ :-2,:]
    b = grid[2:  ,:]
    mask = (X[:,None,:] >= a[None,:,:]) & (X[:,None,:] <  b[None,:,:])
    W = np.sum(y[:, None, None] * mask, axis = 0) 
    W /= n_inputs * np.maximum(np.sum(mask, axis = 0), 1)
    return W


def dynamic_predict(model, init, n_steps):
    """
    Performs a dynamice out-of-sample prediction of a time series

    Args:
        model (object): a prediction model; must implement `predict` method
        init (ndarray): initial sample
        n_steps (int): number of prediction steps

    Returns:
        ndarray: prediction including initial values
    """

    n = len(init)
    y_dynamic = init.tolist()
    for i in range(n_steps):
        x = np.array(y_dynamic[-n:]).reshape(1,-1)
        y = model.predict(x)
        y_dynamic.append(y)
    return np.array(y_dynamic)


def check_solver(solver):
    try:
        solver.fit
        solver.predict
    except:
        raise ValueError("Solver does not implement `fit` or `predict` method, or `coef_` field")


class NeoFuzzyNeuron:  
    """
    Neo-fuzzy neuron model

    Uses sklearn's `Ridge(alpha = 1e-4)` model as a solver. Supports
    uniform and quantile partitioning as well as custom grids

    Attributes:

    """    
    
    def __init__(self, n_rules = None, uniform = True, solver = None, grid = None): 
        if grid is not None:
            self.n_rules = grid.shape[0] - 2
        elif n_rules is not None:
            self.n_rules = n_rules       
        else:
            raise ValueError('Number of rules or grid must be provided')
            
        self.uniform = uniform
        self.weights_ = None
        self.grid = grid.copy() if grid is not None else None
        self.mfunc = get_asym_mfunc(grid) if grid is not None else None
        
        if solver is not None:
            check_solver(solver)
            self.solver_ = deepcopy(solver)
        else:
            self.solver_ = Ridge(alpha = 1e-4)      
                   
    
    def fit(self, X, y): 
        if self.grid is None:
            if self.uniform:
                grid = uniform_grid(X, self.n_rules)
                self.mfunc = get_sym_mfunc(grid)
            else:
                grid = density_grid(X, self.n_rules)
                self.mfunc = get_asym_mfunc(grid)
        
        M = self.mfunc(X)
        M = M.reshape(M.shape[0], M.shape[1] * M.shape[2])

        self.solver_.fit(M, y)
        self.weights_ = self.solver_.coef_.copy()
        self.weights_ = self.weights_.reshape((self.n_rules, X.shape[1]))
        self.c_ = self.solver_.intercept_
        return self

        
    def predict(self, X): 
        M = self.mfunc(X)
        weights = self.weights_[None,:,:]
        return np.sum(weights * M, axis = (1,2)) + self.c_


    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X) 