"""Utility functions for protein classification project."""

import logging
import logging.config
from pathlib import Path
from typing import Union, Optional, Dict, Any
import json
import requests
from Bio.PDB.Structure import Structure  # Import the actual Structure class
from Bio.PDB import PDBParser
import numpy as np
from config.settings import LOGGING, LABEL_SOURCES

def setup_logging(config: Dict[str, Any] = None) -> None:
    """Set up logging configuration.

    Args:
        config: Logging configuration dictionary. If None, uses default from settings.
    """
    if config is None:
        config = LOGGING
    logging.config.dictConfig(config)

def create_directory(directory: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist.

    Args:
        directory: Directory path to create.

    Returns:
        Path object of created directory.
    """
    directory = Path(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    except PermissionError as e:
        logging.error(f"Permission denied when creating directory {directory}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error when creating directory {directory}: {e}")
        raise

def fetch_protein_metadata(uniprot_id: str) -> Dict[str, Any]:
    """Fetch protein metadata from UniProt.

    Args:
        uniprot_id: UniProt identifier.

    Returns:
        Dictionary containing protein metadata.
    """
    if 'uniprot' not in LABEL_SOURCES:
        logging.error("UniProt source not found in LABEL_SOURCES configuration.")
        return {}

    url = f"{LABEL_SOURCES['uniprot']}/{uniprot_id}.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch metadata for {uniprot_id}: {e}")
        return {}

def parse_pdb_structure(pdb_path: Union[str, Path]) -> Optional[Structure]:
    """Parse PDB file into Biopython Structure object.

    Args:
        pdb_path: Path to PDB file.

    Returns:
        Biopython Structure object or None if parsing fails.
    """
    parser = PDBParser(QUIET=True)
    try:
        return parser.get_structure("protein", str(pdb_path))
    except FileNotFoundError:
        logging.error(f"PDB file not found: {pdb_path}")
        return None
    except Exception as e:
        logging.error(f"Failed to parse PDB file {pdb_path}: {e}")
        return None

def calculate_structure_features(structure: Structure) -> Dict[str, float]:
    """Calculate basic structural features of a protein.

    Args:
        structure: Biopython Structure object.

    Returns:
        Dictionary of structural features.
    """
    features = {}
    try:
        # Get all atom coordinates
        coords = np.array([atom.get_coord() for atom in structure.get_atoms()])

        if coords.size == 0:
            logging.warning("Structure contains no atoms.")
            return {"radius_of_gyration": 0.0, "max_dimension": 0.0, "n_residues": 0, "n_atoms": 0}

        # Calculate basic geometric features
        features['radius_of_gyration'] = np.sqrt(np.mean(np.sum((coords - coords.mean(axis=0))**2, axis=1)))
        features['max_dimension'] = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))

        # Count structural elements
        features['n_residues'] = len(list(structure.get_residues()))
        features['n_atoms'] = len(list(structure.get_atoms()))

    except Exception as e:
        logging.error(f"Failed to calculate structure features: {e}")
        return {}

    return features

def save_metadata(metadata: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Save metadata to JSON file.

    Args:
        metadata: Dictionary of metadata.
        output_path: Path to save JSON file.
    """
    output_path = Path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save metadata to {output_path}: {e}")

def load_metadata(metadata_path: Union[str, Path]) -> Dict[str, Any]:
    """Load metadata from JSON file.

    Args:
        metadata_path: Path to JSON file.

    Returns:
        Dictionary of metadata.
    """
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Metadata file not found: {metadata_path}")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON file {metadata_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error loading metadata from {metadata_path}: {e}")
    return {}
