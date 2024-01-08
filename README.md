### Few shot classification using Huggingface / SetFit - Comments on Le Monde
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)



### Things you might re-use (repo guide)
---
`0_lmd_setfit_best_model.ipynb` : train the best performing model. Setfit + LogisticRegression with custom params & optimized hyperparameters.  
`1_optional_lmd_setfit_torch_head.ipynb`: experiment using a differentiable (torch) head.  
`2_optional_lmd_setfitt_XGB_OOM.ipynb`: experiment using a GradientBoosting head and a custom class to prevent out-of-memory errors.  
`3_lmd_setfit_hyperparameters_opti.ipynb`: hyperparameters optimization using Optuna. From +-63% to 69,1% accuracy.  

### Architecture/model experiments, HPO
---

#### Model choice

#### Classification head

#### Hyperparameters optimization


### Deployment - Inference optimization
---

