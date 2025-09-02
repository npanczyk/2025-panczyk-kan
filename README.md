# 2025-panczyk-kan
This repository serves to reproduce the results found in [Opening the AI black-box: Symbolic regression with Kolmogorov-Arnold Networks for advanced energy applications](https://doi.org/10.1016/j.egyai.2025.100595). 

## Paper Citation
Nataly R. Panczyk, Omer F. Erdem, Majdi I. Radaideh,
Opening the AI black-box: Symbolic regression with Kolmogorov-Arnold Networks for advanced energy applications,
Energy and AI,
2025,
100595,
ISSN 2666-5468,
https://doi.org/10.1016/j.egyai.2025.100595.


## Abstract: 
While most modern machine learning methods offer speed and accuracy, few promise interpretability or explainability– two key features necessary for highly sensitive industries, like medicine, finance, and engineering. Using eight datasets representative of one especially sensitive industry, nuclear power, this work compares a traditional feedforward neural network FNN to a Kolmogorov-Arnold Network (KAN). We consider not only model performance and accuracy, but also interpretability through model architecture and explainability through a post-hoc SHapley Additive exPlanations (SHAP) analysis, a game-theory-based feature importance method. In terms of accuracy, we find KANs and FNNs comparable across all datasets, when output dimensionality is limited. KANs, which transform into symbolic equations after training, yield perfectly interpretable models while FNNs remain black-boxes. Finally, using the post-hoc explainability results from Kernel SHAP, we find that KANs learn real, physical relations from experimental data, while FNNs simply produce statistically accurate results. Overall, this analysis finds KANs a promising alternative to traditional machine learning methods, particularly in applications requiring both accuracy and comprehensibility.

**Keywords**: Kolmogorov Arnold Networks; Explainable AI; Interpretable AI; Machine learning; Deep neural networks; Nuclear energy; Critical heat flux


## Getting Started
### Requirements
- python

All necessary packages included in `environment.yml`

### Environment Setup and Activation

```bash
conda env create -f environment.yml

conda activate pykan-env
```

## How to Generate Results

All results reproducible via 

```bash
snakemake -j1
```

Note #1: Due to the the stochastic nature of KANs, the authors have found some unlucky networks to yield NaNs in the test set predictions due to division by zero or imaginary numbers after converting a KAN's B-splines to symbolic expressions. If you encounter such a network, simply re-run the script, the second attempt is unlikely to be unlucky too.

You can re-run any individual step in this workflow via 
```bash
snakemake RULE_NAME_HERE -j1
```
For the KAN and associated equation generation, that would be
```bash
snakemake kan -j1
```

Note #2: We have not provided hyperparameter tuning in the snakemake workflow, but feel free to explore ```workflow/scripts/hypertuning.py``` for a good place to start. 

Note #3: Since the real CHF dataset used in this analysis is not public, we have provided synthetic versions of this dataset in ```data/datasets```. Results using these synthetic data will vary from those presented in the paper.

## License

[MIT](https://choosealicense.com/licenses/mit/)