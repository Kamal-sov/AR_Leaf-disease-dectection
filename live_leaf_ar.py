import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

# Load YOLO model for leaf detection
detector_model_path = 'yolov5_leaf_detector.pt'
leaf_detector = YOLO(detector_model_path)

# Load MobileNetV2 classifier
classifier_model_path = 'model/best_model.pth'
class_names = ['Black Rot', 'Healthy']

mobilenet = models.mobilenet_v2(weights=None)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, len(class_names))
mobilenet.load_state_dict(torch.load(classifier_model_path, map_location=torch.device('cpu')))
mobilenet.eval()

# Preprocessing for classification
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = leaf_detector(frame)
    annotated_frame = frame.copy()
    detected = False

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            img_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            input_tensor = preprocess(img_pil).unsqueeze(0)

            with torch.no_grad():
                output = mobilenet(input_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                prediction_label = f"{class_names[predicted.item()]} - {confidence.item() * 100:.1f}%"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, prediction_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)
            detected = True

    if not detected:
        cv2.putText(annotated_frame, 'None', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Leaf Detection & Classification', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
