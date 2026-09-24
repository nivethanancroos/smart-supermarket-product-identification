"""Module 3: Product classification.

Loads the trained ResNet18 checkpoint and classifies individual product
crops produced by the detection module.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from src import config


class ProductClassifier:
    def __init__(self, model_path: Path = config.MODEL_PATH, classes_path: Path = config.CLASSES_PATH,
                 device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = json.loads(Path(classes_path).read_text())

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])

    def predict(self, image_bgr: np.ndarray) -> tuple[str, float]:
        """Predict a category for a single product crop.

        Returns (label, confidence). If the top confidence is below
        config.UNKNOWN_CONFIDENCE_THRESHOLD, label is UNKNOWN_LABEL instead
        of the (likely wrong) best guess — the model only knows the 25
        trained categories, so out-of-distribution items would otherwise
        always get forced into one of them.
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(probs.argmax())

        confidence = float(probs[idx])
        if confidence < config.UNKNOWN_CONFIDENCE_THRESHOLD:
            return config.UNKNOWN_LABEL, confidence
        return self.classes[idx], confidence
