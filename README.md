# Data-driven concavity engineering for magnetic activation in rigid microspheres
Chang Zhang, Longjun Rao, Lishan Wu, Ruixuan Zhang, Jiazhuan Qin, Meichen Wen, Ke Pei, Hanwen Cheng, Chongyun Liang, Wenbin You*, and Renchao Che*

This set of supplemental materials contains the code and data associated with our paper titled "Data-driven concavity engineering for magnetic activation in rigid microspheres".

## Repo Contents
- [data](./data): summarized data 
- [scripts](./scripts): python codes for summarize the active magnetic domain calculation (Active magnetic domain.py，Screening of active area ratio) and analyze the relationship of concavity, porosity, elements, composition and permeability (machine learning.py)  

# System requirements
## Software Requirements
The codes are tested on Windows10 operating systems and python(v. 3.13.0) jupyter notebook.

# Demo
## Scripts
The codes can be runned by python or python jupyter notebook with stated packages.

`Active magnetic domain angle.py` is used to evaluate the energy consumption of magnetic domain rotation angles and to determine the angles of active magnetic domains in Supplementary Fig.3.
`Screening of active area ratio.py` is used to evaluate the proportion of active magnetic domains under high-frequency magnetic fields in Fig. 1c.
`machine learning.py` is used to optimize XGBoost models for multiple target variables through Optuna-based hyperparameter search and five-fold cross-validation. It provides 1) the optimal hyperparameter combinations for each target, 2) fold-wise evaluation metrics to assess model robustness and consistency in Supplementary Fig.7, and 3) aggregated prediction performance on both cross-validation and independent test sets in Fig. 2c and Supplementary Figs. 11-13. 

 
