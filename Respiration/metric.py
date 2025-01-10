import numpy as np

'''Average level per interval'''
def avg_lvl_per_interval(Amps):
    return np.sum(Amps) / len(Amps)

'''Takes merged sequence as input'''
def reprod_per_field(avg_levels):
    return max(avg_levels) - min(avg_levels)

def mean_reprod_per_fraction(reprods):
    return np.mean(np.array(reprods))

'''Vertical Error'''
def error_per_interval(Amps):
    dt = 0.015
    Times = range(len(Amps))
    slope, _ = np.polyfit(Times, Amps, deg=1)
    duration = dt * ( len(Amps) - 1 )
    return abs(slope) * duration

def stab_per_field(errors):
    return max(errors)

def mean_stab_per_fraction(stabs):
    return np.mean(np.array(stabs))

"""
1 fraction 안에 있는 모든 interval을 고려:
- reprod/stab 계산
- mean/std 계산

Fraction 간의 reprod/stab 추이 분석
"""