# A Reproducible Deep-Learning-Based Computer-Aided Diagnosis Tool for Frontotemporal Dementia Using MONAI and Clinica Frameworks
![Experimental Design](https://github.com/Andreater/FTD_CAD/blob/main/docs/experimental_design.png)
Figure 1. Graphical representation of the main steps of the workflow.

## ToC
1. [Repository description](#repository-description)
2. [Background](#background)
3. [Results](#results)
4. [About the data](#about-the-data)
5. [Scientific paper](#scientific-paper)

## Repository description
This repository contains the code base for the paper "A Reproducible Deep-Learning-Based Computer-Aided Diagnosis Tool for Frontotemporal Dementia Using MONAI and Clinica Frameworks" published in [Life](https://www.mdpi.com/journal/life) is an international, peer-reviewed, open access journal of scientific studies related to fundamental themes in life sciences, from basic to applied research, published monthly online by MDPI. 

## Background
Despite Artificial Intelligence (AI) being a leading technology in biomedical research, real-life implementation of AI-based Computer-Aided Diagnosis (CAD) tools into the clinical setting is still remote due to unstandardized practices during development. Few or no attempts have been made to propose a reproducible CAD development workflow for 3D MRI data. In this paper, we present the development of an easily reproducible and reliable CAD tool using the [Clinica](https://github.com/aramis-lab/clinica) and [MONAI](https://monai.io/) frameworks that were developed to introduce standardized practices in medical imaging. A Deep Learning (DL) algorithm was trained to detect frontotemporal dementia (FTD) on data from the NIFD database to ensure reproducibility (fig.2).

![densenet121](https://github.com/Andreater/FTD_CAD/blob/main/docs/model%20structure.png)
<i>Figure 2. Schematic representation of the DenseNet121 used in this paper.</i>

## Results
The DL model yielded 0.80 accuracy (95% confidence intervals: 0.64, 0.91), 1 sensitivity, 0.6 specificity, 0.83 F1-score, and 0.86 AUC, achieving a comparable performance with other FTD classification approaches (fig.3). [Explainable AI methods](https://github.com/MECLabTUDA/M3d-Cam) were applied to understand AI behavior and to identify regions of the images where the DL model misbehaves (fig.4). 

![classification results](https://github.com/Andreater/FTD_CAD/blob/main/docs/fig2%20new.png)
<i>Figure 3. (A) The confusion matrix indicates classification results for both classes. Rows indicate true labels and columns indicate predicted labels. (B) Receiver Operating Characteristic (ROC) curve of the DenseNet121 classifier obtained when predicting disease status (FTD/NC) using 3D T1w MRI. Area Under the Curve (AUC) was calculated as the definite integral between 0 and 1 on the x-axis and provides an aggregate measure of performance.</i>

Attention maps (fig.4) highlighted that its decision was driven by hallmarking brain areas for FTD and helped us to understand how to improve FTD detection. The proposed standardized methodology could be useful for benchmark comparison in FTD classification. AI-based CAD tools should be developed with the goal of standardizing pipelines, as varying pre-processing and training methods, along with the absence of model behavior explanations, negatively impact regulators’ attitudes towards CAD. The adoption of common best practices for neuroimaging data analysis is a step toward fast evaluation of efficacy and safety of CAD and may accelerate the adoption of AI products in the healthcare system.

![atmaps](https://github.com/Andreater/FTD_CAD/blob/main/docs/atmaps.png)

<i>Figure 4. Coronal view of the brain for one FTD and one NC subject. The original brain scans used for testing are in the upper row, while the attention maps are in the lower row.</i>

## About the data
The DenseNet 121 used here to classify FTD vs. healthy aging subjectes used only 3D T1-weighted Magnetization-Prepared Rapid Acquisition with Gradient Echo (MPRAGE) MRI scans acquired at the first visit. The participants were enrolled by the [frontotemporal lobar degeneration neuroimaging initiative](https://cind.ucsf.edu/research/grants/frontotemporal-lobar-degeneration-neuroimaging-initiative-0), and the dataset is hosted by [the Laboratory of NeuroImaging from the University of Southern California](https://www.loni.usc.edu/). FTD patients included in this database are diagnosed with one of the following disease variants: behavioral variant, semantic variant, progressive non-fluent aphasia, progressive supranuclear palsy, or cortico-basal syndrome. 

## Scientific paper
Termine, A.; Fabrizio, C.; Caltagirone, C.; Petrosini, L.; on behalf of the Frontotemporal Lobar Degeneration Neuroimaging Initiative. A Reproducible Deep-Learning-Based Computer-Aided Diagnosis Tool for Frontotemporal Dementia Using MONAI and Clinica Frameworks. Life 2022, 12, 947. https://doi.org/10.3390/life12070947

[Link to the paper](https://www.mdpi.com/2075-1729/12/7/947)
