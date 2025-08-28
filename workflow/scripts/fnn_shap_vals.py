from fnn import *

def get_fnn_shap(case, dataset, model_path, params_dict, device):
    """Loads dataset and model and calculates kernel shap values. 

    Args:
        models_dict (dict): key = model name (chf, htgr, etc.),
                            values[0] = get_dataset
                            values[1] = model object path
    """
    params = params_dict[case]
    X_train = dataset['train_input'].cpu().detach().numpy()
    X_test = dataset['test_input'].cpu().detach().numpy()
    input_names = dataset['feature_labels']
    output_names = dataset['output_labels']
    save_as =  f"{case.upper()}"
    # feed model args
    input_size = dataset['train_input'].shape[1]
    hidden_nodes = params['hidden_nodes']
    output_size = dataset['train_output'].shape[1]
    model = FNN(input_size, hidden_nodes, output_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    train_samples = dataset['train_input'].shape[0]
    k = int(np.round(0.005*train_samples*input_size))
    if k > 100:
        k = 100
    path = fnn_shap(model, X_train, X_test, input_names, output_names, save_as=save_as, k=k)
    return path

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pymaise_params = {
        'CHF': {
            'hidden_nodes' : [231, 138, 267],
            'num_epochs' : 200,
            'batch_size' : 64,
            'learning_rate' : 0.0009311391232267503,
            'use_dropout': True,
            'dropout_prob': 0.4995897609454529,
        },
        'BWR': {
            'hidden_nodes' : [511, 367, 563, 441, 162],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.0009660778027367906,
            'use_dropout': False,
            'dropout_prob': 0,
        },
        'FP': {
            'hidden_nodes' : [66, 400],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.001,
            'use_dropout': False,
            'dropout_prob': 0,
        },
        'HEAT': {
            'hidden_nodes' : [251, 184, 47],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.0008821712781015931,
            'use_dropout': False,
            'dropout_prob': 0,
        },
        'HTGR': {
            'hidden_nodes' : [199, 400],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.00011376283985074373,
            'use_dropout': True,
            'dropout_prob': 0.3225718287912892,
        },
        'MITR': {
            'hidden_nodes' : [309],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.0008321972582830564,
            'use_dropout': False,
            'dropout_prob': 0,           
        },
        'REA': {
            'hidden_nodes' : [326, 127],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.0009444837105276597,
            'use_dropout': False,
            'dropout_prob': 0,         
        },
        'XS': {
            'hidden_nodes' : [95],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.0003421585453407753,
            'use_dropout': False,
            'dropout_prob': 0,        
        },
        'MITR_A': {
            'hidden_nodes' : [309],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.0008321972582830564,
            'use_dropout': False,
            'dropout_prob': 0,        
        },
        'MITR_B': {
            'hidden_nodes' : [309],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.0008321972582830564,
            'use_dropout': False,
            'dropout_prob': 0,           
        },
        'MITR_C': {
            'hidden_nodes' : [309],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.0008321972582830564,
            'use_dropout': False,
            'dropout_prob': 0,        
        },        
    }
    # get shap values from FNN models, need model path dict from Step 1
    # shap_paths = get_fnn_shap(model_path_dict, pymaise_params, device)
    # print(shap_paths)
    model_paths = snakemake.input.model_paths
    cases = snakemake.params.d_list
    datasets = snakemake.input.datasets

    for case, dataset_file, path in zip(cases, datasets, model_paths):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
        get_fnn_shap(case, dataset, model_path_dict, pymaise_params, device)
