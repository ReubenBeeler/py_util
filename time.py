
import time
from typing import TypeVar, Callable

import numpy as np

_T = TypeVar("_T")

_units_map = {
    'ns': 1.0,
    'μs': 1e-3,
    'ms': 1e-6,
    's': 1e-9,
}

def time_it(units: str, func:Callable[...,_T], *args, **kwargs) -> tuple[float, _T]:
    '''
	Measure the time for a single function invocation of `func`. See the below example. <br>
    .. code-block:: python
		def prime_factor(n: int, sort_ascending: bool): ... # stub
		ret, ms = time_it('ms', prime_factor, 1173, sort_ascending=True)
		print(f'The prime factors of 1173 are {ret} and were computed in {ms} milliseconds.')
    
	:param units: must be one of ['s', 'ms', 'μs', 'ns']
	:type units: str
	:param func: The function to time
	:type func: Callable[..., T]
	:param args: The positional arguments for `func`
	:param kwargs: The keyword arguments for `func`
	:return: A two-element tuple containing 1st the elapsed time and 2nd the return value of the function invocation
	:rtype: tuple[float, T]
	'''
    if not units in _units_map:
        raise ValueError(f"argument 'units' must be one of {[k for k in _units_map.keys()]} but had value {repr(units)}")
    timescale = _units_map.get(units, None)
    start = time.time_ns()
    ret = func(*args, **kwargs)
    end = time.time_ns()
    duration = (end - start)*timescale
    return (duration, ret)