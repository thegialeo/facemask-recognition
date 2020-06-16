# Exoplanet Detection

## Requirements
> pip install -r requirements.txt

## Single Person Classification

### Dataset
For training, we used the following dataset from Kaggle [Mask Datasets V1](https://www.kaggle.com/ahmetfurkandemr/mask-datasets-v1). Put the files into a folder named *dataset* with subfolders *train* and *test*. Each folder *train* and *test* obtain subfolders *mask* and *no_mask*. 

### Usage

### Preprocessing
> python png_to_hdf5.py 
If you have not download the dataset for Multi-Person Object Detection and created the appropriate folder structure yet. Run:
> python png_to_hdf5.py --mode train
> python png_to_hdf5.py --mode test

#### Training

##### From Scratch
> python train.py

##### Finetune
> python train.py --train_mode finetune

#### Testing

## Multi-Person Object Detection

### Dataset
For training, we used the following dataset from Kaggle [Medical Mask Dataset](https://www.kaggle.com/shreyashwaghe/medical-mask-dataset). Put the files into a folder named *dataset/detection* with subfolders *images* and *labels*. Go To subfolder *images* and delete the file *83855_1580055989W0WA.jpg*.

### Usage

#### Preprocessing
If you have not execute png_to_hdf5.py yet, run:
> python png_to_hdf5.py 
Otherwise run:
> python png_to_hdf5.py --mode detection

#### Training

#### Testing 

## Contact
Leo.Nguyen@gmx.de

## License
MIT License





