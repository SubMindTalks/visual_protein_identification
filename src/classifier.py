"""Protein structure classification model and training pipeline."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
        """Initialize the classifier.
        
        Args:
            num_classes: Number of protein classes.
            model_name: Name of the model architecture to use.
            pretrained: Whether to use pretrained weights.
            device: Device to use for training ('cuda' or 'cpu').
        """
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = self._create_model(pretrained)
        self.model.to(self.device)
        
        # Initialize training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=MODEL['learning_rate'],
            weight_decay=MODEL['weight_decay']
        )
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        logger.info(f"Initialized classifier with {model_name} on {self.device}")

    def _create_model(self, pretrained: bool) -> nn.Module:
        """Create the model architecture."""
        try:
            if self.model_name == 'vit_base_patch16_224':
                model = models.vit_b_16(pretrained=pretrained)
                model.heads = nn.Linear(model.hidden_dim, self.num_classes)
            elif self.model_name == 'resnet50':
                model = models.resnet50(pretrained=pretrained)
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            return model
        
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = MODEL['num_epochs'],
              save_dir: Optional[str] = None) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            num_epochs: Number of training epochs.
            save_dir: Directory to save model checkpoints.
            
        Returns:
            Dictionary containing training history.
        """
        if save_dir:
            save_dir = create_directory(save_dir)
        
        logger.info("Starting training...")
        
        try:
            for epoch in range(num_epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                
                for batch_idx, (images, labels, _) in enumerate(train_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                    
                    if batch_idx % 10 == 0:
                        logger.debug(f"Epoch {epoch+1}/{num_epochs} "
                                   f"Batch {batch_idx}/{len(train_loader)} "
                                   f"Loss: {loss.item():.4f}")
                
                avg_train_loss = train_loss / len(train_loader)
                
                # Validation phase
                val_loss, val_accuracy = self.evaluate(val_loader)
                
                # Update history
                self.history['train_loss'].append(avg_train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {avg_train_loss:.4f} - "
                          f"Val Loss: {val_loss:.4f} - "
                          f"Val Accuracy: {val_accuracy:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss and save_dir:
                    self.best_val_loss = val_loss
                    self.save_model(save_dir / 'best_model.pth')
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= MODEL['early_stopping_patience']:
                    logger.info("Early stopping triggered")
                    break
            
            return self.history
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation.
            
        Returns:
            Tuple of (average loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        try:
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
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Make prediction for a single image.
        
        Args:
            image: Input image tensor.
            
        Returns:
            Predicted class probabilities.
        """
        self.model.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device)
            outputs = self.model(image)
            return torch.softmax(outputs, dim=1)

    def save_model(self, path: str) -> None:
        """Save model state."""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'history': self.history
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, path: str) -> None:
        """Load model state."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.history = checkpoint.get('history', self.history)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def analyze_results(self, 
                       test_loader: DataLoader,
                       class_names: List[str],
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Analyze model performance and generate visualizations.
        
        Args:
            test_loader: DataLoader for test data.
            class_names: List of class names.
            output_dir: Directory to save visualizations.
            
        Returns:
            Dictionary containing analysis results.
        """
        if output_dir:
            output_dir = create_directory(output_dir)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        # Collect predictions
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        results = {
            'classification_report': classification_report(
                all_labels, all_preds, target_names=class_names, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
        
        if output_dir:
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(results['confusion_matrix'], 
                       annot=True, 
                       fmt='d',
                       xticklabels=class_names,
                       yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(output_dir / 'confusion_matrix.png')
            plt.close()
            
            # Save results
            with open(output_dir / 'analysis_results.json', 'w') as f:
                json.dump(results, f, indent=2)
        
        return results