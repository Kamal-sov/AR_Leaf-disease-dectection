
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torchvision.models import MobileNet_V2_Weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Paths
train_dir = 'data/train'
val_dir = 'data/val'
model_path = 'model/best_model.pth'

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

# Print class names for debug/logging
print("Detected Classes:", train_dataset.classes)

# Load pretrained MobileNetV2 weights
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(train_dataset.classes))
model = model.to(device)

# Add class weights to loss function
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(train_dataset.targets),
                                     y=train_dataset.targets)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
best_acc = 0.0
for epoch in range(5):  # Adjust number of epochs as needed
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f"GT: {labels.tolist()} | Pred: {predicted.tolist()}")  # Debug prediction print
    acc = correct / total
    print(f'Epoch {epoch+1}, Loss: {running_loss:.4f}, Val Acc: {acc:.4f}')
    
    if acc > best_acc:
        best_acc = acc
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print("Model saved.")

print("Training complete.")
