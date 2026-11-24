"""PCI condition labels used in the CAE rated report (inclusive upper bounds)."""
from bisect import bisect_left

def label(value):
    if not 0 <= value <= 100:
        raise ValueError('PCI must be between 0 and 100.')
    return ('Failed','Serious','Very Poor','Poor','Fair','Satisfactory','Good')[
        bisect_left((10,25,40,55,70,85),value)]

def interval_name(interval):
    lo,hi = (label(interval[key]) for key in ('lower','upper'))
    return lo if lo == hi else f'{lo} to {hi}'
