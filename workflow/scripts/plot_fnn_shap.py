from fnn import *

if __name__=="__main__":
    shap_paths = snakemake.input.shap_vals
    cases = snakemake.params.d_list

    for case, path in zip(cases, shap_paths):
        plot_shap(path, save_as=f'{case}_fnn', type='fnn', width=0.2)
        print_shap(path, save_as=f'{case}', type='fnn')