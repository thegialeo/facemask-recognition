# Facemask Detection

## Requirements
> pip install -r requirements.txt

## Download our trained models
Download the files and put them in a folder named *models*:
[Leo Google Drive](https://drive.google.com/drive/folders/1mNLF0mBMC64I9OA9Diw6tB6N5XYBHOK_?usp=sharing)

## Single Person Classification

### Dataset
For training, we used the following dataset from Kaggle [Mask Datasets V1](https://www.kaggle.com/ahmetfurkandemr/mask-datasets-v1). Put the files into a folder named *dataset* with subfolders *train* and *test*. Each folder *train* and *test* obtain subfolders *mask* and *no_mask*. 

### Preprocessing
> python png_to_hdf5.py
  
If you have not download the dataset for Multi-Person Object Detection and created the appropriate folder structure yet. Run:  
> python png_to_hdf5.py --mode train  
> python png_to_hdf5.py --mode test

### Training SVM
> python SVM.py

### Training MobileNetV2

MobileNetV2 (Sandler et al., 2018)

#### From Scratch
> python train.py

#### Finetune
> python train.py --train_mode finetune

## Multi-Person Object Detection

### Dataset
For training, we used the following dataset from Kaggle [Medical Mask Dataset](https://www.kaggle.com/shreyashwaghe/medical-mask-dataset). Put the files into a folder named *dataset/detection* with subfolders *images* and *labels*. Go To subfolder *dataset/detection/images* and delete the file *83855_1580055989W0WA.jpg*.

### Preprocessing
If you have not execute png_to_hdf5.py yet, run:  
> python png_to_hdf5.py  

Otherwise run:  
> python png_to_hdf5.py --mode detection

### Training

#### Faster-RCNN
Faster-RCNN (Ren et al., 2015)
> python train.py --detection --train_mode faster_rcnn

#### MTCNN
MTCNN (Zhang et al., 2016)
> python train.py --detection --train_mode mtcnn

### Run our algorithm on your image
For the moment, your image has to be in the same folder as run.py:
> python run.py --image_path <path-to-your-image>

## References

Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal NetworksarXiv e-prints, arXiv:1506.01497.  

Sandler, M., Howard, A., Zhu, A., & Chen, L.C. (2018). MobileNetV2: Inverted Residuals and Linear BottlenecksarXiv e-prints, arXiv:1801.04381.  

Szegedy, C., Liu, W., Jia, P., Reed, S., Anguelov, D., Vanhoucke, V., & Rabinovich, A. (2014). Going Deeper with ConvolutionsarXiv e-prints, arXiv:1409.4842.  

Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint Face Detection and Alignment Using Multitask Cascaded Convolutional NetworksIEEE Signal Processing Letters, 23(10), 1499-1503.  

## Contact
Leo.Nguyen@gmx.de

## License
MIT License






