from main import *
from preprocessing import *
import shutil
import os
from kan import *
import datetime as dt
from functools import partial
from plotting import *

# best KAN hyperparameters from hypertuning results
hyperparams = {
    'FP': {
         'depth': 1, 'grid': 8, 'k': 7, 'lamb': 2.0426962767412815e-05, 'lamb_entropy': 5.0346373804560525, 'lr_1': 1.5, 'lr_2': 1.75, 'reg_metric': 'edge_forward_sum', 'steps': 75
         },
    'BWR': {
         'depth': 1, 'grid': 7, 'k': 2, 'lamb': 0.0008912210456241697, 'lamb_entropy': 7.488094627223641, 'lr_1': 1.75, 'lr_2': 1.25, 'reg_metric': 'edge_forward_sum', 'steps': 125
         },
    'HEAT': {
         'depth': 1, 'grid': 7, 'k': 3, 'lamb': 0.00018986520634595234, 'lamb_entropy': 8.209205342922996, 'lr_1': 1.5, 'lr_2': 2, 'reg_metric': 'edge_forward_spline_u', 'steps': 150
         },
    'HTGR': {
         'depth': 1, 'grid': 8, 'k': 3, 'lamb': 1.2166058649376336e-05, 'lamb_entropy': 7.665809430995193, 'lr_1': 0.75, 'lr_2': 1.25, 'reg_metric': 'edge_forward_sum', 'steps': 25
         },
    'MITR_A': {
         'depth': 1, 'grid': 4, 'k': 7, 'lamb': 4.2843417969236176e-05, 'lamb_entropy': 0.025403293360453105, 'lr_1': 1.5, 'lr_2': 1.25, 'reg_metric': 'edge_forward_sum', 'steps': 25
         },
    'MITR_B' : {
         'depth': 1, 'grid': 3, 'k': 2, 'lamb': 1.1115022163426145e-06, 'lamb_entropy': 0.9626840845416629, 'lr_1': 0.75, 'lr_2': 1, 'reg_metric': 'edge_forward_sum', 'steps': 250
         },
    'MITR_C': {
         'depth': 1, 'grid': 3, 'k': 3, 'lamb': 9.739068562318698e-06, 'lamb_entropy': 7.047425350169997, 'lr_1': 1, 'lr_2': 1.5, 'reg_metric': 'edge_forward_spline_n', 'steps': 150
         },
    'MITR': {
         'depth': 1, 'grid': 7, 'k': 6, 'lamb': 0.00033697982852750485, 'lamb_entropy': 5.7732177455173055, 'lr_1': 1.75, 'lr_2': 0.5, 'reg_metric': 'edge_forward_spline_u', 'steps': 150
         },
    'CHF': {
         'depth': 1, 'grid': 9, 'k': 2, 'lamb': 5.123066656699474e-06, 'lamb_entropy': 7.716618914050463, 'lr_1': 2, 'lr_2': 1.75, 'reg_metric': 'edge_forward_spline_u', 'steps': 100
         },
    'REA': {
         'depth': 1, 'grid': 6, 'k': 8, 'lamb': 3.79496703629217e-05, 'lamb_entropy': 0.006504868427044119, 'lr_1': 2, 'lr_2': 0.5, 'reg_metric': 'edge_forward_sum', 'steps': 25
         },
    'XS': {
         'depth': 2, 'grid': 9, 'k': 4, 'lamb': 0.00039029273996368227, 'lamb_entropy': 0.42860645226254324, 'lr_1': 1.25, 'lr_2': 1.25, 'reg_metric': 'edge_forward_spline_u', 'steps': 100
         },
}

def run_model(device, dataset, params, run_name, y_pred_file, y_test_file, lib=None):
    kan = NKAN(dataset, 42, device, params)
    model = kan.get_model()
    spline_metrics = kan.get_metrics(model, run_name)
    expressions, y_pred, y_test = kan.get_equation(model, run_name, simple=0, lib=None)
    plot_pred_v_true(
         y_preds=y_pred[:,0], 
         y_tests= y_test[:,0],
         save_as=run_name, 
         output=dataset['output_labels'][0], 
         color='magenta')
    with open(y_pred_file, "wb") as file:
            pickle.dump(y_pred, file)
    with open(y_test_file, "wb") as file:
            pickle.dump(y_test, file)
    return


if __name__=="__main__":
    if os.path.exists('model'):
        shutil.rmtree("model")
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for case, file, y_pred_file, y_test_file in zip(snakemake.params.d_list, snakemake.input.datasets, snakemake.output.y_preds, snakemake.output.y_tests):
        print(f'NOW RUNNING {case}')
        with open(file, 'rb') as f:
            dataset = pickle.load(f)
        run_model(
            device=device, 
            dataset=dataset, # unpickle
            params=hyperparams[case], 
            run_name=f"{case.upper()}_{str(dt.date.today())}",
            y_pred_file=y_pred_file,
            y_test_file=y_test_file)

