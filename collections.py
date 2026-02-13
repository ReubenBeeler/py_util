from itertools import tee
from typing import Protocol, TypeVar, Generic, Iterable, Iterator, Any

_T = TypeVar("_T")

# # example use case:
# data = [(1, 2), (0.4, 30), (0.7, 17.3), (0.71, 80)] # series of 2D points
# data_split = bin_split(data, 2, (0, 1), key=lambda x:x[0], val=lambda x:x[1])
# for _bin in data_split:
#     _bin[1] = sum(bin[1])
def bin_split(data:Iterable, num_bins:int, start_end:tuple[int|float|None,int|float|None]|None=None, \
              key=lambda x:x, val=lambda x:x, ret_bin_size:bool=False) -> tuple[list[float,list[float]],float]|list[float,list[float]]:
    assert isinstance(num_bins, int) and num_bins > 0

    if isinstance(data, Iterator):
        data = tuple(data)
    assert isinstance(data, Iterable)

    if len(data) == 0: return []

    minx, maxx = min(map(key, data)), max(map(key, data))
    if start_end is None:
        start_end = (minx, maxx)
    else:
        assert isinstance(start_end, Iterable) and len(start_end) == 2
        start_end = list(start_end)
        if start_end[0] is None: start_end[0] = minx
        if start_end[1] is None: start_end[1] = maxx
        start_end = tuple(start_end)
    start, end = start_end
    
    bin_size = (end - start)/num_bins
    assert bin_size > 0.0 , "bad start and end! Negative step size..."

    bins:dict[int,list] = dict()
    for e in data:
        x = key(e)
        if x < start or end < x:
            continue

        bin_index = num_bins - 1 if x == end else int(num_bins * (x - start)/(end - start))
        if bin_index not in bins: bins[bin_index] = []
        bins[bin_index].append(val(e))

    data = [None] * num_bins
    for i in range(num_bins):
        data[i] = [(i + 0.5)*bin_size + start, [] if i not in bins else bins[i]]

    return (data, bin_size) if ret_bin_size else data

def _create_tensor(dims:Iterator[int], default=None):
    dim = next(dims, None)
    if dim is None:
        return default
    else:
        tees = tee(dims, dim)
        return [_create_tensor(tee, default=default) for tee in tees]

def create_tensor(dims:Iterable[int], default=None):
    assert isinstance(dims, Iterable)
    if isinstance(dims, Iterable):
        dims = iter(dims)
    return _create_tensor(dims, default=default)

def _flatten(it:Iterable) -> Iterator:
    for e1 in it:
        if isinstance(e1, Iterable):
            if isinstance(e1, str):
                yield e1
            else:
                for e2 in _flatten(e1):
                    yield e2
        else:
            yield e1

# returns Iterator[_T] if `it` is an Iterator else list[_T]
# TODO add option to manually control if return type is Iterator or list
def flatten(it:Iterable) -> Iterable:
    if isinstance(it, str):
        return it
    iterator = _flatten(it)
    return iterator if isinstance(it, Iterator) else list(iterator)

def buffer(it:Iterable[_T], buffer_length:int) -> list[list[_T]]:
    assert isinstance(buffer_length, int) and buffer_length > 0
    assert isinstance(it, Iterable)
    if isinstance(it, Iterator):
        it = tuple(it) # TODO return Iterator[list[_T]] instead of expanding iterator
    l = ((len(it)-1)//buffer_length) + 1
    return [it[i*buffer_length:(i+1)*buffer_length] for i in range(l)]

class Optional:
    def __init__(self, *args:_T):
        if len(args) > 1: raise AttributeError("only 0 or 1 arg allowed for Optional.__init__")
        self._present = (len(args) == 1)
        if self._present:
            self._t = args[0]
    
    def has(self) -> bool: return self._present
    def empty(self) -> bool: return not self._present
    def get(self) -> _T:
        if self.has(): return self._t
        raise Exception("cannot call get() on an empty optional!")

class Wrapper(Generic[_T]):
    def __init__(self, t:_T):
        self._t = t
    
    def get(self) -> _T:
        return self._t

    def __iter__(self) -> Iterator[_T]:
        yield self._t

# _Chain=TypeVar("_Chain", bound=dict[str,"_Chain"]|list["_Chain"]|str|float|int|None)
# below should be _Chain instead of Any, but python can't handle cyclic generic types
@deprecated("use dotted, jsonpath-ng, or other lib")
def chain(d: Any, *keys:str) -> None|Wrapper[Any]:
    if len(keys) == 0:
        return Wrapper(d)
    key = keys[0]
    if isinstance(d, dict):
        if key in d:
            return chain(d[key], *keys[1:])
        else:
            return None
    elif isinstance(d, list):
        subchains = (chain(e, *keys) for e in d)
        return Wrapper([s.get() for s in subchains if s is not None])
    else:
        return None

def iter_index(iter:Iterable[_T], index:int) -> Wrapper[_T]|None:
    for i, val in enumerate(iter):
        if i == index: return Wrapper(val)
    return None


_T_contra, _T_co = TypeVar("_T_contra"), TypeVar("_T_co")
class SupportsMul(Protocol[_T_contra, _T_co]):
    def __mul__(self, x: _T_contra, /) -> _T_co: ...

# fix these
_MultipliableT1 = TypeVar("_MultipliableT1", bound=SupportsMul[Any, Any])
_MultipliableT2 = TypeVar("_MultipliableT2", bound=SupportsMul[Any, Any])
def prod(iterable: Iterable[_MultipliableT1], /, start: _MultipliableT2) -> _MultipliableT1 | _MultipliableT2:
    ret = start
    for e in iterable:
        ret *= e
    return ret