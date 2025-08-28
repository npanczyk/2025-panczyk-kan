import datetime as dt
# CASES = ['FP']
CASES = ['FP', 'CHF', 'BWR', 'MITR_A', 'MITR_B', 'MITR_C', 'XS', 'HEAT', 'REA', 'HTGR']
# CASES = ['REA', 'HTGR']
case_list = ''.join(CASES)
run_name = 'DEMO'

rule targets:
    input: 
        miter_full = f'{path}{run_name}_params.txt'


rule preprocess:
    input:
        file = 'datasets/chf_train.csv',
    params:
        d_list = CASES,
    output:
        dataset = expand("processed_datasets/{case}.pkl", case=CASES),
    script:
        'preprocessing.py'

# rule hypertune_kan:
#     input: 
#         dataset = f'processed_datasets/{case}.pkl',
#     params:
#         run_name = run_name,
#         max_evals = 1,
#         seed = 42,
#     output:
#         params = f"hyperparameters/{run_name}/{run_name}_params.txt",
#         pruned = f"hyperparameters/{run_name}/{run_name}_pruned.txt",
#         r2 = f"hyperparameters/{run_name}/{run_name}_R2.txt",
#         results = f"hyperparameters/{run_name}/{run_name}_results.txt",
#     script:
#         'hypertuning.py'

rule kan:
    input:
        datasets = expand("processed_datasets/{case}.pkl", case=CASES),
    params:
        d_list = CASES,
    output:
        y_preds = expand('results/Ys/{case}/y_pred_KAN.pkl', case=CASES),
        y_tests = expand('results/Ys/{case}/y_test_KAN.pkl', case=CASES),
    script:
        'run.py'


rule fnn:
    input:
        datasets = expand("processed_datasets/{case}.pkl", case=CASES),
    params:
        d_list = CASES,
    output:
        model_path = expand('models/{case}'+f'__{str(dt.date.today())}.pt', case=CASES),
        y_preds = expand('results/Ys/{case}/y_pred_FNN.pkl', case=CASES),
        y_tests = expand('results/Ys/{case}/y_test_FNN.pkl', case=CASES),
    script:
        'fnn.py'

rule get_fnn_plots:
    input:
        y_preds = expand('results/Ys/{case}/y_pred_FNN.pkl', case=CASES),
        y_tests = expand('results/Ys/{case}/y_test_FNN.pkl', case=CASES),
        datasets = expand("processed_datasets/{case}.pkl", case=CASES),
    params:
        d_list = CASES,
    output:
        plots = expand('figures/pred-v-true/{case}'+f'_FNN_{str(dt.date.today())}.png', case=CASES)
    script:
        'plotting.py'

rule get_stats_scores:
    # dummy input file
    input:
        static_paths = expand("processed_datasets/{case}.pkl", case=CASES), 
    params:
        d_list = CASES,
        trials = 30,
        gpu = "2",
    output:
        kan_scores = f'results/stats/{run_name}/kan_scores.pkl',
        kan_sym_scores = f'results/stats/{run_name}/kan_symbolic_scores.pkl',
        fnn_scores = f'results/stats/{run_name}/fnn_scores.pkl',
    script:
        'stats.py'

rule wilcoxon:
    input:
        kan_scores = f'results/stats/{run_name}/kan_scores.pkl',
        kan_sym_scores = f'results/stats/{run_name}/kan_symbolic_scores.pkl',
        fnn_scores = f'results/stats/{run_name}/fnn_scores.pkl',
    params:
        alpha = 0.05,
        metric = 'R2',
    output:
        kan_wilcoxon = f'results/stats/kan_wilcoxon_R2.tex',
        kan_sym_wilcoxon = f'results/stats/kan_sym_wilcoxon_R2.tex',
    script:
        'wilcoxon.py'