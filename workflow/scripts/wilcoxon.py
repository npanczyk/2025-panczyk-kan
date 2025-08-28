import numpy as np
from scipy.stats import wilcoxon
import pickle
import pandas as pd

def get_wilcoxon(scores1, scores2, metric, alpha, path, type=None):
    results = {}
    for case, output_dict in scores1.items():
        results[case] = []
        std_dict = {}
        for output in output_dict.keys():
            statistic, p_value = wilcoxon(scores1[case][output][metric], scores2[case][output][metric], alternative='two-sided')
            merged = np.array([scores1[case][output][metric], scores2[case][output][metric]])
            std = np.std(merged, axis=0)
            std_dict[output] = std
            
            # check if a significant difference is found
            if p_value < alpha:
                avg1 = np.mean(scores1[case][output][metric])
                avg2 = np.mean(scores2[case][output][metric])
                # if scores1 is better than scores2, yield positive
                if metric == 'R2' and avg1 > avg2:
                    results[case].append('+')
                elif metric == 'R2' and avg1 < avg2: # FNN is better
                    results[case].append('-')
                elif metric == 'MAE' and avg1 < avg2: # reverse the order for error
                    results[case].append('+')
                elif metric == 'MAE' and avg1 > avg2:
                    results[case].append('-')
            else: # no difference between methods
                results[case].append('0')
        std_df = pd.DataFrame(std_dict)
        print( std_df.to_latex(caption=f'Standard deviations for {type} KAN vs FNN on {case} using {metric} as metric.', label=f'{case}_{metric}'))

    for case, vals in results.items():
        results[case] = ''.join(vals)

    df = pd.DataFrame(results, index=['Wilcoxon'])
    # print(df)
    with open(path, 'w') as file:
        file.write(df.to_latex())
    return df

if __name__=="__main__":
    with open(snakemake.input.kan_scores, "rb") as f1:
        kan_dict = pickle.load(f1)

    with open(snakemake.input.kan_sym_scores, "rb") as f2:
        kan_sym_dict = pickle.load(f2)

    with open(snakemake.input.fnn_scores, "rb") as f3:
        fnn_dict = pickle.load(f3)


    kan_results = get_wilcoxon(
        scores1 = kan_dict,
        scores2 = fnn_dict,
        metric = snakemake.params.metric,
        alpha = snakemake.params.alpha,
        path = snakemake.output.kan_wilcoxon,
        type = 'Spline'
    )
    kan_sym_results = get_wilcoxon(
        scores1 = kan_sym_dict,
        scores2 = fnn_dict,
        metric = snakemake.params.metric,
        alpha = snakemake.params.alpha,
        path = snakemake.output.kan_sym_wilcoxon,
        type = 'Symbolic'
    )
