from torchvision.models import resnet18,ResNet18_Weights
import torch
from torch import nn
from torchvision.transforms import ToTensor,Compose,Normalize,Resize
from PIL import Image
import os
from dataset import CustomData
from torch.utils.data import DataLoader

import argparse
def get_args():
    parser = argparse.ArgumentParser(description='build_and_train_resnet18')
    parser.add_argument('--data_path',type=str,default="G:/My Drive/Data/vehicledataset/data/val/Bus/000885_17.jpg")
    parser.add_argument('--output_path',type=str,default='')
    parser.add_argument('--checkpoint_path','-ckp',type=str,default='./Resnet18_torchvisionModels/saved_models/best.pt')


    args = parser.parse_args()
    return args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    print(device)
    model = resnet18(weights = ResNet18_Weights.DEFAULT)
<<<<<<< HEAD
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        ckp = torch.load(args.checkpoint_path)
        model.load_state_dict(ckp['model'])

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features , 5)
=======
    # ckp = torch.load(args.checkpoint_path)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features , 5)
    # model.load_state_dict(ckp['model'])
>>>>>>> d5ee567f41d87a4bbe242d8ad6ba4c7b84e5ed8c
    model.to(device)
    model.eval()
    transform = Compose([
                    Resize((224,224)), 
                    ToTensor(), 
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
    output = os.path.join(args.output_path,'output.txt')
    with open(output, 'w') as f:
        for img_path in os.listdir(args.data_path):
            image_path = os.path.join(args.data_path, img_path)
            if os.path.isfile(image_path):  
                image = Image.open(image_path).convert('RGB')
                image = transform(image)
                image = image.to(device)
                model.eval()
                with torch.no_grad():
<<<<<<< HEAD
                    pred = model(image.unsqueeze(0))  
                prediction = int(torch.max(pred, dim=1).indices)
                print(prediction)
=======
                    pred = model(image.unsqueeze(0))  # Add batch dimension
                prediction = int(torch.max(pred, dim=1).indices)
                f.writelines(f"{prediction}\n")
>>>>>>> d5ee567f41d87a4bbe242d8ad6ba4c7b84e5ed8c
if __name__ == "__main__":
    args = get_args()
    main(args)