import numpy as np

# --- LOCATION ---
def agg_average(x): return np.mean(x)
def agg_median(x): return np.median(x)
def agg_first(x): return x[0]
def agg_last(x): return x[-1]
def agg_central(x): return x[len(x)//2]

# --- DISPERSION ---
def agg_variance(x): return np.var(x)
def agg_std(x): return np.std(x)
def agg_iqr(x): return np.percentile(x, 75) - np.percentile(x, 25)
def agg_range(x): return np.max(x) - np.min(x)

# --- EXTREMA ---
def agg_max(x): return np.max(x)
def agg_min(x): return np.min(x)
def agg_avg_max(x): return np.mean(x) - np.max(x)
def agg_avg_min(x): return np.mean(x) - np.min(x)

# --- ENERGY ---
def agg_energy(x): return np.sum(x**2)
def agg_rms(x): return np.sqrt(np.mean(x**2))

# --- TREND ---
def agg_slope(x):
    t = np.arange(len(x))
    if len(x) < 2: return 0
    cov = np.cov(t, x, bias=True)[0, 1]
    var_t = np.var(t)
    return cov / var_t if var_t > 0 else 0
def agg_endpoint_diff(x): return x[-1] - x[0]

AGG_FUNCS = {
    "Average": agg_average, 
    "Median": agg_median, 
    "First": agg_first,
    "Central": agg_central, 
    "Last": agg_last, 
    "Variance": agg_variance,
    "Std": agg_std, 
    "IQR": agg_iqr, 
    "Max-Min": agg_range,
    "Max": agg_max, 
    "Min": agg_min, 
    "Avg-Max": agg_avg_max,
    "Avg-Min": agg_avg_min, 
    "Energy": agg_energy, 
    "RMS": agg_rms,
    "Slope": agg_slope, 
    "Last-First diff": agg_endpoint_diff,
}

def PAA_reduce(s, w, agg="Average"):
    n = len(s)
    s = np.array(s)
    idx = np.floor(np.linspace(0, w, n, endpoint=False)).astype(int)
    f = AGG_FUNCS[agg]
    res = [f(s[idx == i]) for i in range(w)]
    return np.array(res)