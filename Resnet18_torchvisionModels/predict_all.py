from torch.utils.data import DataLoader
from src.dataset import CustomData
from torchvision.models import resnet18,ResNet18_Weights
import torch
from torch import nn
from torchvision.transforms import ToTensor,Compose,Normalize,Resize
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from build_resnet18_train import ResNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
num_classes = 5
ckp = torch.load('G./saved_models_noPretrained/best.pt')

model = resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features , num_classes)
model.load_state_dict(ckp['model'])

model.to(device)
transform = Compose([
                Resize((224,224)), 
                ToTensor(), 
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

val_data = CustomData(data_dir='././data/test',transform=transform)

val_loader = DataLoader(
    dataset=val_data,
    batch_size=8,
)
total = correct = 0

all_predictions = []
all_labels = []
model.eval()

progress_bar = tqdm(val_loader, colour="yellow")
for (images,labels) in progress_bar:
    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        preds = model(images)
        _, predictions = torch.max(preds, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions,average='macro')
precision = precision_score(all_labels, all_predictions,average='macro')
recall = recall_score(all_labels, all_predictions,average='macro')

print(f'Accuracy: {accuracy}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}')
