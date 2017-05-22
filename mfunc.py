import numpy as np


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
   
    grid = np.vstack(np.linspace(a - h, b + h, n_rules + 3) 
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
        quantiles = np.percentile(X[:,i], q = np.linspace(0, 100, n_rules + 1))
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