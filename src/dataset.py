"""Dataset management and processing for protein structure classification."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import shutil
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config.settings import DATASET
from .utils import create_directory

logger = logging.getLogger(__name__)

class ProteinDataset(Dataset):
    """Dataset for protein structure images with associated metadata."""

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None):
        """Initialize the dataset.

        Args:
            data_root: Root directory containing image data.
            split: Dataset split ('train', 'val', or 'test').
            transform: Optional transforms to apply to images.
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform or self._default_transform()
        self.data_index = self._load_data_index()

        logger.info(f"Initialized {split} dataset with {len(self.data_index)} samples")

    def _default_transform(self) -> transforms.Compose:
        """Create default transformation pipeline."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_data_index(self) -> List[Dict[str, Any]]:
        """Load and validate the dataset index."""
        index_path = self.data_root / f"{self.split}_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Dataset index not found: {index_path}")

        with open(index_path, 'r') as f:
            data_index = json.load(f)

        valid_entries = []
        for entry in data_index:
            image_path = self.data_root / entry['image_path']
            if image_path.exists():
                entry['image_path'] = str(image_path)
                valid_entries.append(entry)
            else:
                logger.warning(f"Image not found: {image_path}")

        return valid_entries

    def __len__(self) -> int:
        return len(self.data_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Get a dataset item."""
        item = self.data_index[idx]

        try:
            # Load and preprocess image
            image = Image.open(item['image_path'])
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = self.transform(image)

            # Convert label to tensor
            label = torch.tensor(item['label'], dtype=torch.long)

            return image, label, item['metadata']

        except UnidentifiedImageError:
            logger.error(f"Unidentified image file at {item['image_path']}")
            raise
        except Exception as e:
            logger.error(f"Error loading item {idx}: {e}")
            raise

class DataManager:
    """Manages dataset organization and splitting."""

    def __init__(self, data_root: str):
        """Initialize data manager.

        Args:
            data_root: Root directory for dataset.
        """
        self.data_root = Path(data_root)
        self.splits = ['train', 'val', 'test']

        # Create split directories
        for split in self.splits:
            create_directory(self.data_root / split)

    def organize_data(self,
                     visualization_results: List[Dict[str, Any]],
                     label_mapping: Dict[str, int]) -> None:
        """Organize rendered images into dataset splits."""
        try:
            # Split data
            train_val_data, test_data = train_test_split(
                visualization_results,
                test_size=DATASET['test_split'],
                random_state=DATASET['random_seed']
            )

            train_data, val_data = train_test_split(
                train_val_data,
                test_size=DATASET['validation_split'] / (1 - DATASET['test_split']),
                random_state=DATASET['random_seed']
            )

            # Process each split
            splits_data = {
                'train': train_data,
                'val': val_data,
                'test': test_data
            }

            for split_name, split_data in splits_data.items():
                self._process_split(split_name, split_data, label_mapping)

        except Exception as e:
            logger.error(f"Failed to organize dataset: {e}")
            raise

    def _process_split(self,
                      split: str,
                      data: List[Dict[str, Any]],
                      label_mapping: Dict[str, int]) -> None:
        """Process a single dataset split."""
        split_dir = self.data_root / split
        index = []

        for result in data:
            pdb_id = result['pdb_id']

            if pdb_id not in label_mapping:
                logger.warning(f"No label found for protein: {pdb_id}")
                continue

            # Process each visualization
            for viz in result.get('visualizations', []):
                src_path = Path(viz['path'])
                if not src_path.exists():
                    logger.warning(f"Visualization file not found: {src_path}")
                    continue

                dst_path = split_dir / src_path.name
                try:
                    # Copy image
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    logger.error(f"Failed to copy {src_path} to {dst_path}: {e}")
                    continue

                # Create index entry
                index.append({
                    'image_path': str(dst_path.relative_to(self.data_root)),
                    'pdb_id': pdb_id,
                    'label': label_mapping[pdb_id],
                    'style': viz.get('style', 'unknown'),
                    'orientation': viz.get('orientation', 'unknown'),
                    'metadata': result.get('metadata', {})
                })

        # Save split index
        index_path = self.data_root / f"{split}_index.json"
        try:
            with open(index_path, 'w') as f:
                json.dump(index, f, indent=2)
            logger.info(f"Created {split} split with {len(index)} images")
        except Exception as e:
            logger.error(f"Failed to save {split} index: {e}")

    def create_dataloaders(self,
                          batch_size: int = 32,
                          num_workers: int = 4) -> Dict[str, DataLoader]:
        """Create DataLoader instances for all splits."""
        dataloaders = {}

        for split in self.splits:
            dataset = ProteinDataset(self.data_root, split=split)
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )

        return dataloaders
