# ResNet18 Training 

This repository contains code for training a ResNet18 model on a specified dataset.

## Data Directory Structure

The data directory structure should look like this:

![image](https://github.com/TienDoan274/Training-Resnet18-with-custom-data/assets/125201131/81904bfa-40e5-4e08-9d14-ef3eac1fa50c)



## Usage

Training the ResNet18 model from torchvisions library with CLI command:


Example:
```
python train.py --data_path ./data --pretrained True --freeze_layers 50 --epochs 20 --batch_size 64
```





