"""Protein structure classification model and training pipeline."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import GradScaler, autocast

from config.settings import MODEL
from .utils import create_directory

logger = logging.getLogger(__name__)

class ProteinClassifier:
    """Handles training and evaluation of the protein classification model."""

    def __init__(self,
                 num_classes: int,
                 model_name: str = MODEL['name'],
                 pretrained: bool = MODEL['pretrained'],
                 device: Optional[str] = None):
        """Initialize the classifier."""
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = self._create_model(pretrained)
        self.model.to(self.device)

        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=MODEL['learning_rate'], weight_decay=MODEL['weight_decay'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5, verbose=True)
        self.scaler = GradScaler()  # Mixed precision training

        # Tracking variables
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

        logger.info(f"Initialized {model_name} on {self.device}")

    def _create_model(self, pretrained: bool) -> nn.Module:
        """Create the model architecture."""
        try:
            if self.model_name == 'resnet50':
                model = models.resnet50(pretrained=pretrained)
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif self.model_name == 'efficientnet_b0':
                model = models.efficientnet_b0(pretrained=pretrained)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, save_dir: Optional[str] = None):
        """Train the model."""
        if save_dir:
            save_dir = create_directory(save_dir)

        logger.info("Starting training...")

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_idx, (images, labels, _) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # Mixed precision training
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            val_loss, val_accuracy = self.evaluate(val_loader)

            # Update history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)

            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # Learning rate scheduler
            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < self.best_val_loss and save_dir:
                self.best_val_loss = val_loss
                self.save_model(save_dir / 'best_model.pth')

        logger.info("Training completed.")
        return self.history

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        return avg_loss, accuracy

    def save_model(self, path: Path) -> None:
        """Save model state."""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'history': self.history,
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def analyze_results(self, test_loader: DataLoader, class_names: List[str], output_dir: Optional[Path] = None):
        """Analyze model performance and generate visualizations."""
        if output_dir:
            output_dir = create_directory(output_dir)

        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        results = {
            'classification_report': classification_report(all_labels, all_preds, target_names=class_names, output_dict=True),
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }

        if output_dir:
            # Save confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d',
                        xticklabels=class_names, yticklabels=class_names, cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(output_dir / 'confusion_matrix.png')
            plt.close()

            with open(output_dir / 'results.json', 'w') as f:
                json.dump(results, f, indent=2)

        return results
