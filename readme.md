# Similarity-based machine learning for small datasets: Predicting bio-lubricant base oil viscosities

## Abstract

Machine learning (ML) has been successfully applied to learn patterns in experimental chemical data to predict molecular properties. However, experimental data can be time consuming and expensive to obtain and, as a result, it is often scarce. Several ML methods face challenges when trained with limited data. Here, we introduce a similarity-based ML approach to efficiently train using small datasets. We group molecules with similar structures, represented by molecular fingerprints, and use these groups to train separate ML models for each group. We apply the methodology to predict kinematic viscosity of bio-lubricant base oil molecules at 40 °C (KV40). We demonstrate the applicability of this method on dynamic viscosity at 25 °C and aqueous solubility. Our method shows noticeable model performance improvement for KV40 prediction compared to transfer learning (TL) and the standard Random Forest (RF). For the other datasets, the performance of similarity-based ML was comparable to the standard RF and both methods outperformed TL. This approach provides a robust framework for limited data that can be readily generalized to a diverse range of molecular datasets. 

* Make sure to unzip the files before running any script in the repository

## Contact
For any questions or issues, please contact Jae Kim (jaekim@udel.edu)