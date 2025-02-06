import numpy as np

"""Reproducibility"""
# Average level per interval
def avg_lvl_per_interval(Amps):
    return np.sum(Amps) / len(Amps)

# Reproducibility (with a list of average levels)
def reproducibility(avg_levels):
    return max(avg_levels) - min(avg_levels)

# Average reproducibility
def mean_reproducibility(reprods):
    return np.mean(np.array(reprods))

"""Stability"""
# Vertical Error
def error_per_interval(Amps):
    dt = 0.015
    Times = [t*dt for t in range(len(Amps))]
    slope, _ = np.polyfit(Times, Amps, deg=1)
    duration = dt * ( len(Amps) - 1 )
    return abs(slope) * duration

# Stability (with a list of vertical distances)
def stability(errors):
    return max(errors)

# Average stability
def mean_stability(stabs):
    return np.mean(np.array(stabs))

"""Statistics"""
def coeff_var(total_metrics): # CV (coefficients of variation)
    """CV_intra & CV_inter"""
    mean = np.mean(total_metrics)
    std = np.std(total_metrics)
    return std / mean