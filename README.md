# DCGAN on Stanford Cars Dataset (dcgan-stanford-cars)
This project trains a Deep Convolutional Generative Adversarial Network (DCGAN) on the Stanford Cars dataset. The code includes preprocessing, model definitions, visualization utilities, and implementation of Differentiable Augmentation for data-efficient GAN training.


## Repository Structure
```
main.ipynb        # Training script for DCGAN
networks.py       # Definitions of generator and discriminator networks
data.py           # Data preprocessing functions
diff_augment.py   # Implementation of differentiable augmentation
viz.py            # Functions for visualizing results
utilities.py      # General utility/helper functions
data/             # Stanford Cars dataset (see instructions below)
results/          # Directory for generated plots and images (created after running main.ipynb)
```


## Setting Up the "data" Folder

The `data` folder should have the following structure:
```
data/
└── stanford_cars/
    ├── cars_test_annos_withlabels.mat
    ├── cars_train/*.jpg
    ├── cars_test/*.jpg
    └── devkit/
        ├── cars_meta.mat
        ├── cars_test_annos.mat
        ├── cars_train_annos.mat
        ├── eval_train.m
        ├── train_perfect_preds.txt
        └── README.txt
```

### Download the Dataset:

Follow the steps below to compose the `data` folder for this project using the approach from 
[thefirebanks](https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616):
1. **Download `cars_train` and `cars_test` folders**
   * Source: [Stanford Cars Dataset (Kaggle)](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)
   * Extract these folders into the following structure:
     ```
     data/stanford_cars/cars_train
     data/stanford_cars/cars_test
     ```
2. **Download additional metadata files**
   * Files: `cars_meta.mat`, `car_train_annos.mat`, `car_test_annos.mat`, `eval_train.m`, `train_perfect_preds.txt` and `README.txt`
   * Source: [car_devkit.tgz](https://github.com/pytorch/vision/files/11644847/car_devkit.tgz)
   * Extract these files into the `data/stanford_cars/devkit/` folder:
     ```
     data/stanford_cars/devkit/
     ```
3. **Download `cars_test_annos_withlabels.mat`**
   * Source: [PyTorch-StanfordCars-Classification (Kaggle)](https://www.kaggle.com/code/subhangaupadhaya/pytorch-stanfordcars-classification)
   * Place the file in the `data/stanford_cars/` folder:
     ```
     data/stanford_cars/cars_test_annos_withlabels.mat
     ```


## Differentiable Augmentation

This project incorporates the Differentiable Augmentation technique introduced in the paper ["Differentiable Augmentation for Data-Efficient GAN Training"](https://arxiv.org/abs/2006.10738).
