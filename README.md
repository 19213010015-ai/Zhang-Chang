# Data-driven concavity engineering for magnetic activation in rigid microspheres
Chang Zhang, Longjun Rao, Lishan Wu, Ruixuan Zhang, Jiazhuan Qin, Meichen Wen, Ke Pei, Hanwen Cheng, Chongyun Liang, Wenbin You*, and Renchao Che*

This set of supplemental materials contains the code and data associated with our paper titled "Data-driven concavity engineering for magnetic activation in rigid microspheres".

## Repo Contents
- [data](./data): summarized data calculated from Micromagnetic simulation
- [scripts](./scripts): python codes for summarize the overall HT-DFT calculation (HT_DFT_surfE.py) and analyze the relationship between host-guest atomic property and THH preference (machine_learning.py)  机器学习五折运算，活跃磁畴筛选 ，

# System requirements
## Software Requirements
The codes are tested on Windows10 operating systems and python(v. 3.13.0) jupyter notebook.

# Demo
## Scripts
The codes can be runned by python or python jupyter notebook with stated packages

`HT_DFT_surfE.py` creates 1) host-guest nanoparticle surface energy plot in Fig S5., 2) surface energy difference heatmap in Fig 1., and 3) preference heatmap in Fig S6.
`machine_learning.py` is used for feature importance analysis by forest classifier, shap analysis, and gaussian process classifier as shown in Fig 2.
