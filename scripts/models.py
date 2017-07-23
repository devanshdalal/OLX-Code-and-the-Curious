
import time
from functools import wraps

run_times = {}
def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        if function.__name__ not in run_times:
        	run_times[function.__name__]=0
        run_times[function.__name__]+=(t1-t0)
        return result
    return function_timer