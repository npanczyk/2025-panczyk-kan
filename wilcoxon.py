import numpy as np
from scipy.stats import wilcoxon
import pickle
import pandas as pd

def get_wilcoxon(scores1, scores2, metric, alpha, path):
    results = {}
    for case, output_dict in scores1.items():
        results[case] = []
        for output in output_dict.keys():
            statistic, p_value = wilcoxon(scores1[case][output][metric], scores2[case][output][metric], alternative='two-sided')
            print(statistic, p_value)

            # NEED MAJDI TO CHECK THIS LOGIC!!!!
            if p_value < alpha and statistic > 0:
                results[case].append('+')
                print('+')
            elif p_value < alpha and statistic < 0:
                results[case].append('-')
                print('-')
            else:
                results[case].append('0')

    for case, vals in results.items():
        results[case] = ''.join(vals)

    df = pd.DataFrame(results, index=['Wilcoxon'])
    print(df)
    with open(path, 'w') as file:
        file.write(df.to_latex())
    return df

if __name__=="__main__":
    with open(snakemake.input.kan_scores, "rb") as f1:
        kan_dict = pickle.load(f1)

    print(kan_dict)

    with open(snakemake.input.kan_sym_scores, "rb") as f2:
        kan_sym_dict = pickle.load(f2)
    with open(snakemake.input.fnn_scores, "rb") as f3:
        fnn_dict = pickle.load(f3)
    print(fnn_dict)

    kan_results = get_wilcoxon(
        scores1 = kan_dict,
        scores2 = fnn_dict,
        metric = snakemake.params.metric,
        alpha = snakemake.params.alpha,
        path = snakemake.output.kan_wilcoxon
    )
    kan_sym_results = get_wilcoxon(
        scores1 = kan_sym_dict,
        scores2 = fnn_dict,
        metric = snakemake.params.metric,
        alpha = snakemake.params.alpha,
        path = snakemake.output.kan_sym_wilcoxon
    )
