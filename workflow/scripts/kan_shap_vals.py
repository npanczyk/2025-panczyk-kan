from explainability import *

def get_kan_shap(dataset, equation_file, case):
    X_test = dataset['test_input'].cpu().detach().numpy()
    X_train = dataset['train_input'].cpu().detach().numpy()
    input_names = dataset['feature_labels']
    output_names = dataset['output_labels']
    train_samples = X_train.shape[0]
    input_size = X_train.shape[1]
    k = int(np.round(0.005*train_samples*input_size))
    if k > 100:
        k = 100
    path = kan_shap(equation_file, X_train, X_test, input_names, output_names, save_as=case, k=k, width=0.2)
    return path

if __name__=="__main__":
    cases = snakemake.params.d_list
    datasets = snakemake.input.datasets
    equations = snakemake.input.equations
    for dataset_file, equation, case in zip(datasets, equations, cases):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
        get_kan_shap(dataset, equation, case)

