import numpy as np

"""Reproducibility"""
# Average level per interval
def avg_lvl_per_interval(Amps):
    return np.sum(Amps) / len(Amps)

# Field reproducibility
def reprod_per_field(avg_levels):
    return max(avg_levels) - min(avg_levels)

# Average reproducibility per fraction
def mean_reprod_per_fraction(reprods):
    return np.mean(np.array(reprods))

"""Stability"""
# Vertical Error
def error_per_interval(Amps):
    dt = 0.015
    Times = [t*dt for t in range(len(Amps))]
    slope, _ = np.polyfit(Times, Amps, deg=1)
    duration = dt * ( len(Amps) - 1 )
    return abs(slope) * duration

# Field stability
def stab_per_field(errors):
    return max(errors)

# Average stability per fraction
def mean_stab_per_fraction(stabs):
    return np.mean(np.array(stabs))

"""Statistics"""
def coeff_var(total_metrics): # CV (coefficients of variation)
    """CV_intra & CV_inter"""
    mean = np.mean(total_metrics)
    std = np.std(total_metrics)
    return std / mean