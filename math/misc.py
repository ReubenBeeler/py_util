from typing import Iterable
import math

float_regex_str = r"(?:\+|-)?(?:\d+\.?\d*|\d*\.?\d+)(?:[Ee](?:\+|-)?\d+)?"

_digits = tuple(str(i) for i in range(10))
_nzdigits = tuple(str(i) for i in range(1, 10))
# This kinda sucks lol
def sci_notation(f:float, precision:int) -> str:
    assert isinstance(f, float)
    assert isinstance(precision, int) and precision > 0
    s = str(f)
    if math.isinf(f) or math.isnan(f):
        return s
    # if n == 0:
    #     ind_fnlzd = sum(1 for c in s if c not in _nzdigits)
    #     ret = s[:ind_fnlzd]
    #     if ind_fnlzd > 0 and ret[-1] == '.':
    #         ret = ret[:-1]
    #     digits = sum(1 for c in ret if c in _digits)
    #     return ret + (f"e{s.index('.')-digits}" if '.' not in ret else '')
    arr = [None] * len(s)
    ind = count = 0
    for j in range(len(s)):
        arr[ind] = c = s[j]
        ind += 1
        if c in _digits and not (count == 0 and c == '0'):
            count += 1
            if count >= precision:
                break

    return "".join(arr[:ind]) + (precision-count)*"0" + (f"e{s.index('.')-ind}" if s.index('.') >= ind else "")

def calculate_robust_statistics(data:Iterable[float]):
  bias = np.median(data)
  abs_devs = [abs(e-bias) for e in data]
  med_abs_dev = np.median(abs_devs)
  robust_error = math.sqrt(bias**2 + (1.48*med_abs_dev)**2)
  return bias, robust_error
# n1 / n2 but handles nan or +/- inf

def divide(n1, n2):
    if isinstance(n1, np.ndarray):
        if isinstance(n2, np.ndarray):
            assert n1.shape[0] == n2.shape[0]
            return np.array([divide(e1/e2)] for e1,e2 in zip(n1, n2))
        else:
            assert isinstance(n2, (float,int))
            return np.array([divide(e1/n2)] for e1 in n1)
    else:
        assert isinstance(n1, (float,int))
        assert isinstance(n2, (float,int))
    if n2 != 0: return n1/n2
    if n1 == 0: return math.nan
    return math.copysign(math.inf, n1)