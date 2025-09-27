# Deep_Learning Project (by Hyunjong Kim)

## **1. Pneumonia Classification using X-ray images (Kaggle Open Dataset)**
* Purpose
    * X-ray is rather less expensive and more accessible among the medical imaging techniques.
    * Pneumonia is a global pandemic that has tremendous number of patients.
    * Therefore, diagnosing pneumonia using X-ray would largely help the global health.
    * This projects further aims to classify the pneumonia group into bacterial class and viral class.
* Dataset
    * Format: JPG file
    * Datasize: 5193 patients
* Model
    * Ensemble of 3 DL models
        * 1. ResNet50
        * 2. VGG-19
        * 3. EfficientNet

## **2. Risk Prediction of Hepatocellular Carcinoma based on Longitudinal Data (KNUH)**
* Purpose
    * It is known that Hepatitis C patients have higer risk developing to HCC.
    * This project aims to investigate how much risk the HCV patients possess developing to HCC.
* Dataset
    * Description: Longitudinal table data (eg. Albumin, Bilirubin, ...)
    * Format: CSV file
    * Datasize: 2872 patients
* Model
    * 1. Light Gradient Boosting Machine
    * 2. Decision Tree
    * 3. Random Forest

## **3. Classification of Sjogren's Syndrome using US images (DCMC)**
* Purpose
    * By now, diagnosing Sjogren's Syndrome (aka. SJS) mainly rely on blood test rather than imaging technologies.
    * However, blood test is costly while US imaging is relatively inexpensive.
    * Hence, it will ease patients' worries if SJS can be diagnosed with US images.
* Dataset
    * Format: JPG file
    * Image Size: (224, 224) -> Grayscale
* Model
    * ResNet50

## **4. Analysis of Respiration during radiotherapies (YUMC)**
* Purpose
    * Maintaining constant respiration during radiotherapies is crucial.
    * Respiration skills are known to be trainable.
    * This project aims to identify that the prognosis of trained-patients was positive compared to untrained ones.
* Dataset
    * Description: Time series data of patient's ventral movement (Anterior-Posterior)
    * Format: Text file