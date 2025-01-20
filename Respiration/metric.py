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
def R_squared(total_metrics):
    slope, intercept = np.polyfit(range(1, len(total_metrics)+1), total_metrics, deg=1)
    fitted_metrics = slope * ( range(1, len(total_metrics)+1) ) + intercept
    mean = np.mean(total_metrics)
    SST = np.sum( (total_metrics - mean) ** 2 )
    SSR = np.sum( (total_metrics - fitted_metrics) **2 )
    return 1 - SSR/SST

def p_value(total_metrics):
    return

def coeff_var(total_metrics): # CV (coefficients of variation)
    """CV_intra & CV_inter"""
    mean = np.mean(total_metrics)
    std = np.std(total_metrics)
    return std / mean

def ICC(total_metrics): # Intraclass Correlation Coefficient
    return