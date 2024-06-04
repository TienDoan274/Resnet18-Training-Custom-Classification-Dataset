# ResNet18 Training 

This repository contains code for training a ResNet18 model on a specified dataset.

## Data Directory Structure

The data directory structure should look like this:

```
.
└── data/
    ├── test/
    ├── train/
    └── val
        ├── image1.jpg
        ├── image2.jpg
        ├── image3.jpg
        ├── ...
```

## Installation
```
git clone https://github.com/TienDoan274/Resnet18_training.git
pip install -r requirements.txt
```

## Usage
```
cd .\Resnet18_training\Resnet18_ginorchvisionModels\
```
### Training
Training the ResNet18 model from torchvisions library with CLI command:


Example:
```
python train.py --data_path 'path_to_your_data_folder/data' --pretrained True --freeze_layers 50 --epochs 20 --batch_size 64
```

### Inference
To run inference using the trained model, use the following command:

Example:
```
python predict.py --data_path 'path_to_your_images_folder/images' --checkpoint_path 'saved_models/best.pt' 
```


