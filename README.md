# HH-modeling


Code and toy data to reproduce the results in *Hierarchical Dynamic Modeling for Individualized Bayesian Forecasting* (https://arxiv.org/abs/2101.03408).


Modeling occurs at the following levels:
- Return
- Total Spend
- Spend in Category
- Spend in Subcategory
- Item Quantity

## Data

Simulated data for three household groups:
- HH Group 1: high spending and purchasing households (``HH1-DATA.csv``)
- HH Group 2: moderate spending and purchasing households (``HH2-DATA.csv``)
- HH Group 3: low spending and purchasing households (``HH3-DATA.csv``)
200 Households in each group, data for 8 items over 110 weeks 

## Code

- ``EDA.ipynb``: notebook to reproduce EDA plots and tables, plotting individual households and items
- ``HHmodels.py``: script to run all models, except for simultaneous modeling.  All covariates (including simultaneous covariates) treated as known
    - p(Return): Bernoulli DGLM
    - p(log total spend | Return): DLM
    - Category level: DLMM
    - Sub-Category level: DLMM
    - Item level and direct modeling: DCMM
- ``HHanalysis.py``: script to analyze results of ``HHmodels.py`` for all modeling levels
    - p(Return)
    - p(log total spend | Return)
    - Category level
    - Sub-Category level
    - Item level
    - Simultaneous modeling
    - Direct modeling
    - Metric functions for MAD, MAPE and ZAPE, calibration and coverage
    - Plotting functions for individual price sensitive households
- ``HHsimultaneous.py``: simultaneous level modeling, starting from global level, p(Return), use mean or median point forecasts from higher levels in modeling hierarchy to condition out no returns and as covariates at lower levels of modeling 

