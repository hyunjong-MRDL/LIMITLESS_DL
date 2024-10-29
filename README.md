# Deep_Learning Project (by Hyunjong Kim)

## **1. Autism Spectrum Disorder (Classfication)**
* Purpose
    * Classify ASD patients from normal class
* Dataset
    * Description: 3D MR brain images
    * Format: Nifti file
    * Image Size: (256, 256, 256)
* Model
    * 3D-ResNet

## **2. DL aided CT reconstruction (Canon)**
* Purpose
    * Canon developed a DL aided CT reconstruction algorithm.
    * It helps reconstruction with higher resolution, which can help better diagnosis.
    * This project aims to confirm the validity of this new technique.
* Dataset
    * Description: RT structures
    * Format: DICOM file
* Method
    * Compare the IoU & Dice coefficient of images reconstructed from previous technology and the new one.

## **3. Machine Learning based Risk Prediction of Hepatocellular Carcinoma (HCC)**
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

## **4. DL based Pneumonia Classification using X-ray images**
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

## **5. Solving Jigsaw Puzzle using Deep Learning Model**
* Dataset
    * Format: JPG file
    * Image Size: (3, 224, 224) -> RGB
* Model:
    * Vision Transformer

## **6. DL based Classification of Sjogren's Syndrome using Ultrasound images**
* Purpose
    * By now, diagnosing Sjogren's Syndrome (aka. SJS) mainly rely on blood test rather than imaging technologies.
    * However, blood test is costly while US imaging is relatively inexpensive.
    * Hence, it will ease patients' worries if SJS can be diagnosed with US images.
* Dataset
    * Format: JPG file
    * Image Size: (224, 224) -> Grayscale
* Model
    * ResNet50