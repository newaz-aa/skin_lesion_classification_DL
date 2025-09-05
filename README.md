# skin_lesion_classification_DL
Deep learning framework for multiclass skin lesion classification for mobile-acquired images. It includes a curated dataset (70+ categories), CNN and Transformer models, and Grad–CAM–based interpretability.


## Overview

This repository contains the implementation for our research on skin lesion classification using deep learning models. The models are trained on smartphone-acquired non-dermoscopic skin lesion images, aiming to make automated dermatological screening more accessible and practical.

We benchmarked several state-of-the-art CNN and Transformer-based models (e.g., ResNet, EfficientNet, Swin Transformer) and provided the full pipeline for preprocessing, training, and evaluation. The results, model interpretation all are included in this repository.


## Libraries Used

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.14-red)](https://keras.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-darkred)](https://pytorch.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12.2-green)](https://seaborn.pydata.org/)
[![LIME](https://img.shields.io/badge/LIME-0.2-yellow)](https://github.com/marcotcr/lime)

## Dataset
We curated a dataset of more than 27,000 images spanning over 50 skin disease categories. Images were collected from diverse online sources through extensive searches to capture a broad range of skin tones and lesion types. We also identified several public repositories that provide images for specific categories of skin conditions. These already published public repositories are cited below. Not all retrieved images were of sufficient quality, and low-resolution or unclear samples were excluded during the curation process. The dataset will be made public upon acceptance of the paper.

1. PAD-UFES-20: Contains 2,298 images for 6 different categories (https://data.mendeley.com/datasets/zr7vgbcyr2/1)
2. Monkeypox Skin Images Dataset (MSID): 3 categories - Monkeypox, Chickenpox, Measles (https://data.mendeley.com/datasets/r9bfpnvyxr/6)
3. Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0): 5 categories (https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20)
4. Dermatology Atlas: rare disease conditions available (https://www.atlasdermatologico.com.br/)

   

## DL Models Deployed

[![VGG16](https://img.shields.io/badge/VGG16-ImageNet-orange)](https://keras.io/api/applications/vgg/#vgg16-function)
[![InceptionV3](https://img.shields.io/badge/InceptionV3-ImageNet-blue)](https://keras.io/api/applications/inceptionv3/)
[![EfficientNetB0](https://img.shields.io/badge/EfficientNetB0-ImageNet-green)](https://keras.io/api/applications/efficientnet/)
[![Xception](https://img.shields.io/badge/Xception-ImageNet-purple)](https://keras.io/api/applications/xception/)
[![MobileNetV2](https://img.shields.io/badge/MobileNetV2-ImageNet-red)](https://keras.io/api/applications/mobilenet/#mobilenetv2-function)
[![ViT](https://img.shields.io/badge/ViT-PyTorch-lightblue)](https://github.com/facebookresearch/deit)
[![Swin Transformer](https://img.shields.io/badge/Swin-Transformer-lightgreen)](https://github.com/microsoft/Swin-Transformer)
[![DeiT](https://img.shields.io/badge/DeiT-PyTorch-yellow)](https://github.com/facebookresearch/deit)

## Results
The Swin-base transformer achieved the highest accuracy of more than 80%.
![App Screenshot](https://github.com/newaz-aa/skin_lesion_classification_DL/blob/main/Confusion_matrices/swin_base%20confusion_matrix.png)


## Training 

![App Screenshot](https://github.com/newaz-aa/skin_lesion_classification_DL/blob/main/Training_Loss_curves/efficientnet_training_curves.png)

## Grad-CAM

![App Screenshot](https://github.com/newaz-aa/skin_lesion_classification_DL/blob/main/Grad-CAM/gradcam_subplot_keloid.jpg)

## Prediction Analysis

![App Screenshot](https://github.com/newaz-aa/skin_lesion_classification_DL/blob/main/Prediction_analysis/GridImage.png)

## Citation
pending - waiting for arxiv submission completion

## Acknowledgement
We sincerely thank the creators and maintainers of the publicly available datasets used in this study:  

- **PAD-UFES-20**: [Mendeley Dataset](https://data.mendeley.com/datasets/zr7vgbcyr2/1)  
- **Monkeypox Skin Images Dataset (MSID)**: [Mendeley Dataset](https://data.mendeley.com/datasets/r9bfpnvyxr/6)  
- **Mpox Skin Lesion Dataset v2.0 (MSLD v2.0)**: [Kaggle Dataset](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20)  
- **Dermatology Atlas**: [Atlas Dermatológico](https://www.atlasdermatologico.com.br/)  
 
