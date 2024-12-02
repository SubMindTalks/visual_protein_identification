"""Protein visualization module using Py3Dmol."""

import logging
from pathlib import Path
from typing import Optional, List
import py3Dmol
from Bio.PDB import PDBParser, PDBIO
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from .utils import create_directory

logger = logging.getLogger(__name__)

class VisualizationError(Exception):
    """Custom exception for visualization-related errors."""
    pass

class ProteinVisualizer:
    """Handles protein visualization and image generation using Py3Dmol."""
    
    ORIENTATIONS = {
        'front': {'rotation': [0, 0, 0]},
        'back': {'rotation': [0, 180, 0]},
        'top': {'rotation': [-90, 0, 0]},
        'bottom': {'rotation': [90, 0, 0]},
        'left': {'rotation': [0, 90, 0]},
        'right': {'rotation': [0, -90, 0]}
    }
    
    REPRESENTATION_TYPES = {
        'spheres': lambda v: v.setStyle({'sphere': {'radius': 1.0}}),
        'surface': lambda v: v.setStyle({'surface': {'opacity': 0.8}}),
        'mesh': lambda v: v.setStyle({'wireframe': {'color': 'grey'}}),
        'sticks': lambda v: v.setStyle({'stick': {}}),
        'ribbon': lambda v: v.setStyle({'cartoon': {'style': 'ribbon'}}),
        'cartoon': lambda v: v.setStyle({'cartoon': {}}),
        'wireframe': lambda v: v.setStyle({'line': {}}),
        'spacefill': lambda v: v.setStyle({'sphere': {'radius': 1.6}})
    }
    
    def __init__(self, output_dir: str, width: int = 1000, height: int = 1000):
        """Initialize visualizer and set up output directory.
        
        Args:
            output_dir: Directory to save rendered images.
            width: Image width in pixels.
            height: Image height in pixels.
        """
        self.output_dir = create_directory(output_dir)
        self.width = width
        self.height = height
        self.parser = PDBParser(QUIET=True)
        logger.info(f"Initialized visualizer with output directory: {output_dir}")

    def _create_viewer(self) -> py3Dmol.view:
        """Create a new Py3Dmol viewer instance."""
        try:
            viewer = py3Dmol.view(width=self.width, height=self.height)
            viewer.removeAllModels()
            viewer.setBackgroundColor('white')
            return viewer
        except Exception as e:
            logger.error(f"Failed to create viewer: {e}")
            raise VisualizationError(f"Failed to create viewer: {e}")

    def process_pdb(self, pdb_path: str, metadata: dict = None) -> None:
        """Process a single PDB file and generate all visualization types.
        
        Args:
            pdb_path: Path to PDB file.
        
        Raises:
            FileNotFoundError: If PDB file doesn't exist.
            VisualizationError: If visualization fails.
        """
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            logger.error(f"PDB file not found: {pdb_path}")
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        
        try:
            logger.info(f"Processing PDB file: {pdb_path}")
            pdb_name = pdb_path.stem
            
            # Load PDB file content
            with open(pdb_path, 'r') as f:
                pdb_content = f.read()
            
            for rep_type in self.REPRESENTATION_TYPES:
                for orientation in self.ORIENTATIONS:
                    self._render_view(pdb_content, pdb_name, rep_type, orientation)
            
            logger.info(f"Successfully processed PDB file: {pdb_path}")
        except Exception as e:
            logger.error(f"Failed to process PDB file {pdb_path}: {e}")
            raise VisualizationError(f"Failed to process PDB file: {e}")

    def _render_view(self, pdb_content: str, pdb_name: str, rep_type: str, orientation: str) -> None:
        """Render a specific view of the protein.
        
        Args:
            pdb_content: Content of PDB file.
            pdb_name: Name of the PDB file (without extension).
            rep_type: Type of representation to render.
            orientation: Viewing orientation.
        
        Raises:
            VisualizationError: If rendering fails.
        """
        try:
            logger.debug(f"Rendering {rep_type} view from {orientation} for {pdb_name}")
            
            # Create new viewer for each render
            viewer = self._create_viewer()
            
            # Add model and apply style
            viewer.addModel(pdb_content, "pdb")
            self.REPRESENTATION_TYPES[rep_type](viewer)
            
            # Set orientation
            rotation = self.ORIENTATIONS[orientation]['rotation']
            viewer.setView(viewergrid=(0, 0, 0, *rotation))
            viewer.zoomTo()
            
            # Render and save image
            filename = f"{rep_type}_{pdb_name}_{orientation}.tiff"
            output_path = self.output_dir / filename
            
            # Get PNG data and convert to TIFF
            png_data = viewer.png()
            png_bytes = base64.b64decode(png_data)
            
            with Image.open(BytesIO(png_bytes)) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Save as TIFF with high quality
                img.save(output_path, format='TIFF', quality=100)
            
            logger.debug(f"Successfully saved image to {output_path}")
        except Exception as e:
            logger.error(f"Failed to render view: {e}")
            raise VisualizationError(f"Failed to render view: {e}")

    @staticmethod
    def get_center_of_mass(structure) -> np.ndarray:
        """Calculate center of mass of protein structure.
        
        Args:
            structure: Biopython Structure object.
            
        Returns:
            numpy array of [x, y, z] coordinates.
        """
        coords = []
        masses = []
        
        for atom in structure.get_atoms():
            coords.append(atom.get_coord())
            masses.append(atom.mass)
            
        coords = np.array(coords)
        masses = np.array(masses)
        total_mass = masses.sum()
        
        return (coords * masses[:, np.newaxis]).sum(axis=0) / total_mass