
from typing import *

import numpy as np

from ..time import divide
from ..collections import flatten

# _Num = TypeVar("float", bound=float|int)
_Func = Callable[[list[float]], float]

def _fd_derivative(i:int, coords:list[float], func:_Func, step:float) -> float:
    '''
	Computes the 1D derivative of a multivariate function `func` using finite differences.
	
	:param i: index of which coordinate to take the derivative
	:type i: int
	:param coords: The coordinates at which to compute the derivative.
	:type coords: list[float]
	:param func: The function whose derivative is to be computed.
	:type func: Callable[[list[float]], float]
	:param step: The step size for finite difference using the following convention (for a 1D derivative): (f(x + step) - f(x - step))/(2 * step).
	:type step: float
	:return: Returns the derivative
	:rtype: float
	'''
    c_old = coords[i]
    coords[i] = c_old + step; f_p = func(coords)
    coords[i] = c_old - step; f_m = func(coords)
    coords[i] = c_old
    return (f_p - f_m)/(2*step)

def fd_gradient(coords:list[float], func:_Func, step:float) -> list[float]:
    '''
	Computes the gradient of `func` using finite differences.
	
	:param coords: The coordinates at which to compute the gradient.
	:type coords: list[float]
	:param func: The function whose gradient is to be computed.
	:type func: Callable[[list[float]], float]
	:param step: The step size for finite difference using the following convention (for a 1D derivative): (f(x + step) - f(x - step))/(2 * step).
	:type step: float
	:return: Returns the gradient as an array
	:rtype: list[float]
	'''
    l = len(coords)
    def g_i(i:int):
        return _fd_derivative(i, coords, func, step)
    return [g_i(i) for i in range(l)]

# TODO use dynamic programming and caching
def fd_hessian(coords:list[float], func:_Func, step:float, analytic_gradient_i_func:Callable[[int],_Func]|None=None) -> list[list[float]]:
    '''
	Computes the hessian of `func` using finite differences.
	
	:param coords: The coordinates at which to compute the hessian.
	:type coords: list[float]
	:param func: The function whose hessian is to be computed.  
	:type func: Callable[[list[float]], float]
	:param step: The step size for finite difference using the following convention (for a 1D derivative): (f(x + step) - f(x - step))/(2 * step).
	:type step: float
	:param analytic_gradient_i_func: Let `analytic_gradient_i_func=None` for calculating the finite-difference hessian over the finite-difference gradient. For calculating finite-difference hessian over the analytic gradient, set analytic_gradient_i_func(i: int) to be a function that returns the `i`th component of the analytic gradient.
	:type analytic_gradient_i_func: Callable[[int], _Func] | None
	:return: Returns the hessian as a 2D array
	:rtype: list[list[float]]
	
    '''
    l = len(coords)
    def g_i_func(i:int) -> _Func:
        return lambda coords: _fd_derivative(i, coords, func, step)
    g_i_func = g_i_func if analytic_gradient_i_func is None else analytic_gradient_i_func
    def h_ij_func(i:int, j:int) -> _Func:
        return lambda coords: _fd_derivative(j, coords, g_i_func(i), step)
    
    return [[h_ij_func(i, j)(coords) for j in range(l)] for i in range(l)]

def check_derivatives(coords:Iterable, func:Callable, gradient:Callable, hessian:Callable, step:float, rel_tol:float, abs_tol:float, verbosity:int=1, print:Callable[[str],None]=print):
    '''
	Quick and easy function for analyzing how close the finite-difference approximation is to the analytic solution of the gradient and hessian.
	'''
    g, h = gradient(coords), hessian(coords)
    fd_g = fd_gradient(coords, func, step)
    fd_h = fd_hessian(coords, func, step)#, analytic_g_i_func=lambda i: lambda coords: grad(coords)[i])

    def _check(tag:str, fds, analytics):
        rel_errs = [divide(fd - a, a) for fd, a in zip(fds, analytics)]
        abs_errs = [fd - a for fd, a in zip(fds, analytics)]
        rel_outside_abs = (divide(e, a) for e,a in zip(abs_errs, analytics) if abs(e) > abs_tol)
        outside = sum(1 for rel_err in rel_outside_abs if abs(rel_err) > rel_tol)
        if outside == 0:
            if verbosity >= 3:
                print(f"{tag} check passed!")
        if outside > 0 or verbosity >= 4:
            err_thresholds = {"rel": rel_tol, "abs": abs_tol}
            if verbosity >= 1:
                print(f"{outside}/{len(abs_errs)} {tag} components are outside both error thresholds ({err_thresholds})!")
            if verbosity >= 2:
                print(f"----- {tag[0]} check -----")
                print(f"max abs |error|:\t{max(abs_errs, key=abs)}")
                print(f"avg abs |error|:\t{np.mean([abs(e) for e in abs_errs])}")
                print(f"std abs |error|:\t{np.std(abs_errs)}")

        return {"rel": rel_errs, "abs": abs_errs}
    
    g_errs = _check("Gradient", fd_g, g)
    h_errs = _check("Hessian", flatten(fd_h), flatten(h))

    import pdb; pdb.set_trace()

    return g_errs, h_errs