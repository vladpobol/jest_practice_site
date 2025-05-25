import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import os
import json
import torch.nn as nn

class FlowerClassifier:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize model with pretrained weights
        self.model = models.efficientnet_v2_s(pretrained=False)
        # Replace the classifier head
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=102, bias=True)
        # Load the trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Update transform to match training configuration
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(384,384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load class names from JSON
        self.class_names = self._load_class_names()
    
    def _load_class_names(self):
        labels_path = os.path.join(os.path.dirname(__file__), 'labels.json')
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        return [labels[str(i)] for i in range(102)]
    
    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, top_class = torch.topk(probabilities, 5)
            
        results = []
        for prob, class_idx in zip(top_prob, top_class):
            results.append({
                "class": self.class_names[class_idx.item()],
                "probability": prob.item()
            })
        
        return results 