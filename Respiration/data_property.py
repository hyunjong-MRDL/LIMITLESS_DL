from scipy.stats import shapiro, levene, bartlett
from scipy.stats import mannwhitneyu

def run_wilcoxon_rank_sum(group1, group2, alternative='two-sided'):
    stat, p = mannwhitneyu(group1, group2, alternative=alternative)
    return stat, p

def check_normality_and_homoscedasticity(merged_data, use_bartlett=True, alpha=0.05):
    result = {}

    group1, group2 = merged_data[0], merged_data[1]
    stat1, p1 = shapiro(group1)
    stat2, p2 = shapiro(group2)

    result["group1_normality_p"] = p1
    result["group2_normality_p"] = p2
    result["normality"] = (p1 > alpha) and (p2 > alpha)

    if result["normality"]:
        if use_bartlett:
            stat_var, p_var = bartlett(group1, group2)
            method = "Bartlett"
        else:
            stat_var, p_var = levene(group1, group2)
            method = "Levene"

        result["equal_variance_method"] = method
        result["equal_variance_p"] = p_var
        result["equal_variance"] = p_var > alpha
    else:
        result["equal_variance_method"] = None
        result["equal_variance_p"] = None
        result["equal_variance"] = None

    return result