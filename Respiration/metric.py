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

# 전체 Time series에 대한 그래프를 그리면서
# 그 위에 reproducibility와 stability를
# 같이 plotting할 수 있을까? (그래프 부분적으로 그리기)