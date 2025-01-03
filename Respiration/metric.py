import numpy as np

'''Reproducibility'''
def average_level(Amps):
    return np.sum(Amps) / len(Amps)

'''Takes merged sequence as input'''
def calc_reprod(avg_levels):
    return max(avg_levels) - min(avg_levels)

'''Stability'''
def maximum_change(Times, Amps):
    slope, _ = np.polyfit(Times, Amps, deg=1)
    duration = Times[len(Times)-1] - Times[0]
    return abs(slope) * duration

def calc_stab(max_changes):
    return max(max_changes)

# 전체 Time series에 대한 그래프를 그리면서
# 그 위에 reproducibility와 stability를
# 같이 plotting할 수 있을까? (그래프 부분적으로 그리기)