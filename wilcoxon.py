import numpy as np
from scipy.stats import wilcoxon

def get_wilcoxon(scores1, scores2, alpha=0.05):
    results = {}
    for case, output_dict in scores1.items():
        results[case] = []
        for output in output_dict.keys():
            statistic, p_value = wilcoxon(scores1[case][output], scores2[case][output], alternative='two-sided')
            if p_value < alpha and statistic > 0:
                results[case].append('+')
            elif p_value < alpha and statistic < 0:
                results[case].append('-')
            else:
                results[case].append('0')
    return results

if __name__=="__main__":
    