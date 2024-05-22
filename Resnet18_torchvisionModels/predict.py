from torchvision.models import resnet18,ResNet18_Weights
import torch
from torch import nn
from torchvision.transforms import ToTensor,Compose,Normalize,Resize
from PIL import Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
model = resnet18(weights = ResNet18_Weights.DEFAULT)
ckp = torch.load('./saved_models/best.pt')
num_features = model.fc.in_features
model.fc = nn.Linear(num_features , 5)
model.load_state_dict(ckp['model'])
model.to(device)
model.eval()
transform = Compose([
                ToTensor(), 
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
img_path = os.path.join('././data/test/Truck/003490_17.jpg')
image = Image.open(img_path).convert('RGB')
image = transform(image)[None,:,:,:].to(device)
pred = model(image)

print(torch.max(pred,dim = 1).indices)