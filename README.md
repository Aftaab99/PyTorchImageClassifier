# PyTorch image classifier for the notMNIST dataset
The notMNIST dataset contains 28x28px images of letters A to J in different fonts.
This repository aims to serve as a sample for image classification in pytorch. It includes
1. Custom Dataset and dataloaders
2. Training and testing the model
3. Creating a multiclass image classification model

## Dataset
![A](https://github.com/Aftaab99/PyTorch-CNN-for-notMNIST-dataset/blob/master/Dataset/Train/A/QWNoaWxsZXNCbHVyTGlnaHQtRXh0ZW5kZWQub3Rm.png)
![B](https://github.com/Aftaab99/PyTorch-CNN-for-notMNIST-dataset/blob/master/Dataset/Train/B/QW1lcmljYW5UeXBld3JpdGVyQ29uQlEtQm9sZC5vdGY%3D.png)
![C](https://github.com/Aftaab99/PyTorch-CNN-for-notMNIST-dataset/blob/master/Dataset/Test/C/Q2FyZGluYWwgUmVndWxhci50dGY%3D.png)
![D](https://github.com/Aftaab99/PyTorch-CNN-for-notMNIST-dataset/blob/master/Dataset/Train/D/QmxldyBFeHRlbmRlZCBJdGFsaWMudHRm.png)
![E](https://github.com/Aftaab99/PyTorch-CNN-for-notMNIST-dataset/blob/master/Dataset/Test/E/MjAwcHJvb2Ztb29uc2hpbmUgcmVtaXgudHRm.png)
![F](https://github.com/Aftaab99/PyTorch-CNN-for-notMNIST-dataset/blob/master/Dataset/Test/F/MDRiXzA4LnR0Zg%3D%3D.png)
![G](https://github.com/Aftaab99/PyTorch-CNN-for-notMNIST-dataset/blob/master/Dataset/Test/G/Q2FsdmVydE1ULm90Zg%3D%3D.png)
![H](https://github.com/Aftaab99/PyTorch-CNN-for-notMNIST-dataset/blob/master/Dataset/Train/H/Q29yb25ldC1TZW1pQm9sZC1JdGFsaWMgRXgudHRm.png)
![I](https://github.com/Aftaab99/PyTorch-CNN-for-notMNIST-dataset/blob/master/Dataset/Train/I/Q2FlY2lsaWFMVFN0ZC1MaWdodC5vdGY%3D.png)
![J](https://github.com/Aftaab99/PyTorch-CNN-for-notMNIST-dataset/blob/master/Dataset/Train/J/MlRvb24gU2hhZG93LnR0Zg%3D%3D.png)

The dataset contains over 18500 images for training and 469 images for testing. All images are grayscale 28x28px images.

## Model used and accuracy
A convolutional neural network achieves an accuracy of 92.37% on this dataset.