
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2

# Define class names (adjust based on your dataset)
class_names = ['Black Rot', 'Healthy']  # Replace or extend as needed

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load('model/best_model.pth', map_location=device))
model.eval().to(device)

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load and preprocess image
image_path = 'data/sample_leaf.jpg'  # Replace with actual test image path
img = Image.open(image_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    pred_class = class_names[pred_idx]
    confidence = probs[0][pred_idx].item()

# Load original image for display
img_cv = cv2.imread(image_path)
cv2.rectangle(img_cv, (20, 20), (500, 80), (0, 255, 0), -1)
text = f"{pred_class}: {confidence * 100:.2f}%"
cv2.putText(img_cv, text, (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

# Show image
cv2.imshow('Disease Detection - AR Overlay', img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
