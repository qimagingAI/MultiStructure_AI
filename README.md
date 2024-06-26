# MultiStructure_AI

This repository contains the code for the NLST multi-organ Segmentation with Artificial Intelligence project.

Results of this project were published in RSNA Radiology: Michalowska, Buchwald, Shanbhag, Bednarski, Killekar, Miller, Builoff, Lemley, Berman, Dey, and Slomka, 2024 -- _AI for multi-structure incidental findings and mortality prediction on chest CT_.

Summary Statement: A fully automated artificial intelligence model integrates multi-structure segmentation, feature extraction, and quantification methods from chest CT scans to identify high-risk extrapulmonary features and predict mortality.  

Key Results:  

* In this retrospective study of 24,401 patients undergoing chest CT for lung cancer screening, the AI model predicted 10-year all-cause mortality with an AUC of 0.72 (p<0.001). 

* The model achieved an AUC of 0.71 for predicting 2-year all-cause mortality, with mortality rates of 1.13% above the 0.3% below the risk threshold. 

* For predicting significant extrapulmonary incidental findings, the model demonstrated an AUC of 0.70, with a detection prevalence of 16% among patients.  

Files in this repository:

 - `xgboost_code.py` -- code used to create and validate the AI model described in the study;
 - `env_linux.yaml` -- Anaconda Python environment for unix-based systems;
 - `env_windows.yaml` -- Anaconda Python environment for MS Windows system.

The code used to perform the analyzes described in the Radiology article depends also on the following open source repositories:

The TotalSegmentator code is publicly available (https://github.com/wasserth/TotalSegmentator), cLSTM code is publicly available under a Creative Commons BY-NC license at https://doi.org/10.5281/zenodo.10632288, PyRadiomics code is available at https://pyradiomics.readthedocs.io/en/latest/, and the XGBoost Python implementation is available at https://xgboost.readthedocs.io/en/stable/python/. 

---

Graphical abstract of the article:

![images/Figure 1_Revision.png](images/Figure%201_Revision.png)


---