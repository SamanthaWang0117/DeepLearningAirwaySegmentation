# DeepLearningAirwaySegmentation

## Dataset
[Find the raw segmentation data here](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucapanl_ucl_ac_uk/EvfbOxYzN6RJgcaLF6OmWLkBZQxHC13FXUiYF5ERX3oybA?e=xgsLI8)

Please create a folder DRIVE_2 in your local machine under DeepLearingAirwaySegmentation


Select data from the preprocessed dataset.  

Save datasets under directory DRIVE_2. 

[Find training dataset here](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucapanl_ucl_ac_uk/EvfbOxYzN6RJgcaLF6OmWLkBZQxHC13FXUiYF5ERX3oybA?e=g7b3Tz)

[Find testing dataset here](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucapanl_ucl_ac_uk/EmVLLwHm_5FBgFOU3vPG5YEB69IoDOVXjn8PHg--HOM-mQ?e=eHItWw)





## Prerequisites
__python__   3.8.10

__scikit-fmm__   2021.7.8

__scikit-image__ 0.18.1

__pytorch__ 1.7.1

__torchvision__ 0.8.2

__numpy__ 1.21.2

__matplotlib__ 3.3.4

__tqdm__ 4.61.2

## CNN Training


Run this code under DeepLearningAirwaySegmentation directory for training the CNN model
```python
python3 train_cnn.py
``` 

## Vertexs and edges extraction

Run this code for extracting graph features
```python
python3 script.py
``` 

## VGN Training
```python
python3 train_VGN.py
``` 

## Result
CNN probability map result will be shown in folder train_img and test_img under DRIVE_2 directory

VGN probability map result will be shown in folder VGN_results under DRIVE_2 directory
