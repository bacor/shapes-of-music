import numpy as np
from itertools import product
from scipy.stats import entropy
from scipy.optimize import differential_evolution


def iter_param_grid(param_grid):
    """Iterate a parameter grid, yielding dictionaries with parameter values.

    >>> param_grid = dict(a=(0, 10, 2), b=(0, 90, 3))
    >>> params = list(iter_param_grid(param_grid))
    >>> params[0]
    {'a': 0.0, 'b': 0.0}
    >>> [list(p.values()) for p in params]
    [[0.0, 0.0], [0.0, 45.0], [0.0, 90.0], [10.0, 0.0], [10.0, 45.0], [10.0, 90.0]]

    You can also pass numpy arrays with the paramater values directly:

    >>> param_grid = dict(a=[0, 5], b=(0, 10, 2))
    >>> params = list(iter_param_grid(param_grid))
    >>> params
    [{'a': 0, 'b': 0.0}, {'a': 0, 'b': 10.0}, {'a': 5, 'b': 0.0}, {'a': 5, 'b': 10.0}]

    Parameters
    ----------
    param_grid : dictionary
        A dictionary with parameter names as keys and as values either numpy
        arrays of valus, of a triplet of (v_min, v_max, n_steps)

    Yields
    -------
    dict
        A dictionary with the current value of each parameter
    """
    for param, grid in param_grid.items():
        if type(grid) == tuple:
            if not len(grid) == 3:
                raise ValueError(
                    f"Invalid grid for parameter {param}: expected a tuple "
                    f"of length 3, but found one of length {len(grid)}."
                )
            else:
                v_min, v_max, n_steps = grid
                param_grid[param] = np.linspace(v_min, v_max, n_steps)
        elif type(grid) == list or type(grid) == np.ndarray:
            pass
        else:
            raise ValueError(
                f"Invalid grid for parameter {param}: either pass a tuple "
                f"of length 3, or a numpy array."
            )
    param_names = param_grid.keys()
    for param_values in product(*param_grid.values()):
        params = dict(zip(param_names, param_values))
        yield params


def entropy_score(typology, contours):
    freqs = typology.type_dist(contours)
    score = entropy(freqs, base=2)
    return dict(score=score, freqs=freqs)


class TypologyOptimizer:
    trace = []

    def __init__(self, typology, scoring_fn=entropy_score, **kwargs):
        """Optimizer for typologies.

        Parameters
        ----------
        typology : Typology
            A typology class (not an instance!)
        scoring_fn : callable, optional
            An alternative scoring function that accepts two arguments:
            the typology and a collection of contours. By default, the entropy
            of the type distribution is used. The function should return a
            dictionary with a 'score'. Other information in the dictionary is
            stored in the trace.
        """
        self.typology_class = typology
        self.kwargs = kwargs
        self.scoring_fn = scoring_fn

    def score(self, contours, params):
        typology = self.typology_class(**params, **self.kwargs)
        result = self.scoring_fn(typology, contours)
        if type(result) == dict:
            if not "score" in result:
                raise Exception(
                    "The output of the scoring function should be a dictionary "
                    "containing at least a score."
                )
        else:
            result = dict(score=result)
        result["params"] = params
        self.trace.append(result)
        return result

    def grid_search(self, contours, param_grid):
        """Perform a grid search over a parameter grid, optimizing the entropy
        of the type distribution of the typology. You can specify another scoring
        function if desired.

        All results are stored in the .trace attribute of the optimizer.

        Parameters
        ----------
        contours : iterable
            An iterable of contours
        param_grid : dict
            A parameter grid; see iter_param_grid for details

        Returns
        -------
        dict, float
            A dictionary with the optimal parameters and the best score
        """
        self.trace = []
        best = dict(score=-np.inf, freqs=None, params=None)
        for params in iter_param_grid(param_grid):
            result = self.score(contours, params)
            if result["score"] > best["score"]:
                best = result
        return best["params"], best["score"]

    def loss_function(self, contours, param_names):
        """Return a function that can be minimized by scipy.optimize.

        Here's an example:

        >>> from scipy.optimize import basinhopping
        >>> from .contour import Contour
        >>> from .huron import HuronTypology
        >>> optimizer = TypologyOptimizer(HuronTypology)
        >>> contours = [Contour([3,2,1]), Contour([1,2,3])]
        >>> loss = optimizer.loss_function(contours, ['tolerance'])
        >>> res = basinhopping(loss, [0], seed=0)
        >>> res.fun
        -1.0
        >>> res.x
        array([0.])

        Parameters
        ----------
        contours : iterable
            An iterable of contours
        param_names : list
            A list of parameter names (strings)
        """

        def loss(param_values):
            params = dict(zip(param_names, param_values))
            result = self.score(contours, params)
            return -1 * result["score"]

        return loss

    def optimize(self, contours, param_bounds, **kwargs):
        """Optimize the typology using scipy.optimize.differential_evolution.

        Note that the score will be maximized (so we minimize -score).

        Parameters
        ----------
        contours : list
            An iterable of contour
        param_bounds : dict
            A dictionary with the parameter bounds
        **kwargs
            Keyword arguments will be passed to scipy.optimize.minimize

        Returns
        -------
        scipy.optimize.OptimizeResult
            See scipy.optimize.minimize
        """
        self.trace = []
        param_names = param_bounds.keys()
        bounds = list(param_bounds.values())
        func = self.loss_function(contours, param_names)
        res = differential_evolution(func, bounds, **kwargs)
        best = dict(zip(param_names, res.x))
        return best, res


if __name__ == "__main__":
    import doctest

    doctest.testmod()
