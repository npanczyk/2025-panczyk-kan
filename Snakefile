import datetime as dt

CASES = ['FP', 'CHF', 'BWR', 'MITR_A', 'MITR_B', 'MITR_C', 'XS', 'HEAT', 'REA', 'HTGR']

import os
RAW_DATA_FILES = expand("data/datasets/{inpfile}", inpfile=os.listdir("data/datasets")
)

run_name = 'DEMO'

rule all:
    input:
        # kan_y_preds = expand('results/Ys/{case}/y_pred_KAN.pkl', case=CASES),
        # fnn_y_preds = expand('results/Ys/{case}/y_pred_FNN.pkl', case=CASES),
        fnn_plots = expand('results/figures/pred-v-true/{case}'+f'_FNN.png', case=CASES),
        # kan_stats_scores = f'results/stats/{run_name}/kan_scores.pkl',
        kan_wilcoxon_scores = f'results/stats/kan_wilcoxon_R2.tex',
        kan_shap_plots = expand("results/figures/shap/{case}_kan.png", case=CASES),
        fnn_shap_plots = expand("results/figures/shap/{case}_fnn.png", case=CASES),
        dag = "dag.png"

rule preprocess:
    input:
        RAW_DATA_FILES,
    params:
        d_list = CASES,
    output:
        dataset = expand("data/processed_datasets/{case}.pkl", case=CASES),
    script:
        'workflow/scripts/preprocessing.py'

rule kan:
    input:
        datasets = expand("data/processed_datasets/{case}.pkl", case=CASES),
    params:
        d_list = CASES,
    output:
        y_preds = expand('results/Ys/{case}/y_pred_KAN.pkl', case=CASES),
        y_tests = expand('results/Ys/{case}/y_test_KAN.pkl', case=CASES),
        equations = expand("results/equations/{case}.txt", case=CASES),
    script:
        'workflow/scripts/run.py'

rule fnn:
    input:
        datasets = expand("data/processed_datasets/{case}.pkl", case=CASES),
    params:
        d_list = CASES,
    output:
        model_path = expand('fnn-models/{case}.pt', case=CASES),
        y_preds = expand('results/Ys/{case}/y_pred_FNN.pkl', case=CASES),
        y_tests = expand('results/Ys/{case}/y_test_FNN.pkl', case=CASES),
    script:
        'workflow/scripts/fnn.py'

rule get_fnn_plots:
    input:
        y_preds = expand('results/Ys/{case}/y_pred_FNN.pkl', case=CASES),
        y_tests = expand('results/Ys/{case}/y_test_FNN.pkl', case=CASES),
        datasets = expand("data/processed_datasets/{case}.pkl", case=CASES),
    params:
        d_list = CASES,
    output:
        plots = expand('results/figures/pred-v-true/{case}'+f'_FNN.png', case=CASES)
    script:
        'workflow/scripts/plotting.py'

rule get_stats_scores:
    # dummy input file
    input:
        raw_data = RAW_DATA_FILES,
        datasets = expand("data/processed_datasets/{case}.pkl", case=CASES), 
    params:
        d_list = CASES,
        trials = 30,
        gpu = "2",
    output:
        kan_scores = f'results/stats/{run_name}/kan_scores.pkl',
        kan_sym_scores = f'results/stats/{run_name}/kan_symbolic_scores.pkl',
        fnn_scores = f'results/stats/{run_name}/fnn_scores.pkl',
    script:
        'workflow/scripts/stats.py'

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
        'workflow/scripts/wilcoxon.py'

rule kan_shap_values:
    input:
        datasets = expand("data/processed_datasets/{case}.pkl", case=CASES),
        equations = expand("results/equations/{case}.txt", case=CASES),
    params:
        d_list = CASES,
    output:
        shap_vals = expand("results/shap-values/{case}_kan.pkl", case=CASES),
    script:
        'workflow/scripts/shap_vals.py'

rule kan_shap_plot:
    input:
        shap_vals = expand("results/shap-values/{case}_kan.pkl", case=CASES),
    params:
        d_list = CASES,
    output:
        plots = expand("results/figures/shap/{case}_kan.png", case=CASES),
    script:
        'workflow/scripts/explainability.py'

rule fnn_shap_values:
    input:
        model_paths = expand('fnn-models/{case}.pt', case=CASES), 
        datasets = expand("data/processed_datasets/{case}.pkl", case=CASES),
    params:
        d_list = CASES,
    output:
        shap_vals = expand("results/shap-values/{case}_fnn.pkl", case=CASES),
    script:
        'workflow/scripts/fnn_shap_vals.py'

rule fnn_shap_plot:
    input:
        shap_vals = expand("results/shap-values/{case}_fnn.pkl", case=CASES),
    params:
        d_list = CASES,
    output:
        plots = expand("results/figures/shap/{case}_fnn.png", case=CASES),
    script:
        'workflow/scripts/plot_fnn_shap.py'

rule build_dag:
    input: "Snakefile"
    output:
        "dag.png"
    shell:
        "snakemake --dag | dot -Tpng > {output}"