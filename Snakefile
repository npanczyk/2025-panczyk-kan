import datetime as dt
CASES = ['CHF', 'BWR', 'MITR_A', 'MITR_B', 'MITR_C', 'MITR', 'XS', 'FP', 'HEAT', 'REA', 'HTGR']
# run_name = f'{case}_{str(dt.date.today())}'

# rule targets:
#     input: 
#         fig_file = "figures.log",
#         miter_full = f'{path}{run_name}_params.txt'

# rule preprocess:
#     input:
#         file = 'datasets/chf_train.csv',
#     output:
#         dataset = f'processed_datasets/{case}.pkl',
#     script:
#         'preprocessing.py'

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

# rule hypertune_fnn:
#     input:



