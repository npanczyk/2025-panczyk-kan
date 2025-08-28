import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_feature_importances(importances, labels):
    fig, ax = plt.subplots(figsize =(8, 10))
    x_vals = np.arange(0, len(labels))
    ax.bar(x_vals, importances, width=0.4)
    ax.set_ylabel('Relative Feature Importances')
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(labels, rotation=45)
    return fig

def plot_overfitting(n_params, train_rmse, test_rmse, cont_train_rmse, cont_test_rmse, save_as):
    fig, ax1 = plt.subplots()
    ax1.plot(n_params, train_rmse, marker="o")
    ax1.plot(n_params, test_rmse, marker="o")
    ax1.legend(['train', 'test'], loc="lower left")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Parameters')
    ax1.set_ylabel('RMSE')
    fig.show()
    fig.savefig(f'figures/{save_as}_nparams.png', dpi=300)

    fig, ax2 = plt.subplots()
    ax2.plot(cont_train_rmse, linestyle='dashed')
    ax2.plot(cont_test_rmse)
    ax2.legend(['train', 'test'])
    ax2.set_xlabel('Step')
    ax2.set_ylabel('RMSE')
    ax2.set_yscale('log')
    fig.show()
    fig.savefig(f'figures/{save_as}.png', dpi=300)
    return fig

def plot_pred_v_true(y_preds, y_tests, save_as, output, color='magenta'):
    fig, ax = plt.subplots()
    ax.scatter(y_preds, y_tests, color=color)
    ax.plot(y_tests, y_tests, color='black', label=f'Hypothetical Perfect Prediction for {output}')
    ax.legend()
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.savefig(save_as, dpi=300)

if __name__=="__main__":
    for case, y_preds_file, y_tests_file, dataset_file, plot in zip(
        snakemake.params.d_list,
        snakemake.input.y_preds,
        snakemake.input.y_tests,
        snakemake.input.datasets,
        snakemake.output.plots):
        with open(dataset_file, 'rb') as f1:
            dataset = pickle.load(f1)
        with open(y_preds_file, 'rb') as f2:
            y_preds = pickle.load(f2)
        with open(y_tests_file, 'rb') as f3:
            y_tests = pickle.load(f3)
        plot_pred_v_true(
            y_preds=y_preds[:,0],
            y_tests=y_tests[:,0],
            save_as=plot,
            output=dataset['output_labels'][0],
            color='skyblue'
        )
