# HH-modeling


Code and toy data to reproduce the results in *Hierarchical Dynamic Modeling for Individualized Bayesian Forecasting* (https://arxiv.org/abs/2101.03408).


Modeling occurs at the following levels:
- Return
- Total Spend
- Spend in Category
- Spend in Subcategory
- Item Quantity

**HHmodels.py** - fits models at each level listed above where simultaneous covariates are treated as **known**.  Also models directly at the Item level for comparison without using the modeling decomposition

**HHsimultaneous.py** - fits models at all levels and simultaneous covariates are forecasted values from higher levels in the model.
