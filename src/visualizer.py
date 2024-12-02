"""Protein visualization module using Py3Dmol."""

import logging
from pathlib import Path
from typing import Optional, Dict, List
import py3Dmol
from Bio.PDB import PDBParser
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
        'spheres': {'sphere': {'radius': 1.0}},
        'surface': {'surface': {'opacity': 0.8}},
        'mesh': {'wireframe': {'color': 'grey'}},
        'sticks': {'stick': {}},
        'ribbon': {'cartoon': {'style': 'ribbon'}},
        'cartoon': {'cartoon': {}},
        'wireframe': {'line': {}},
        'spacefill': {'sphere': {'radius': 1.6}}
    }

    def __init__(self, output_dir: str, width: int = 800, height: int = 800):
        """Initialize visualizer and set up output directory."""
        self.output_dir = create_directory(output_dir)
        self.width = width
        self.height = height
        self.parser = PDBParser(QUIET=True)
        self.viewer = None
        logger.info(f"Initialized visualizer with output directory: {output_dir}")

    def _initialize_viewer(self) -> None:
        """Initialize or reinitialize the Py3Dmol viewer."""
        try:
            self.viewer = py3Dmol.view(width=self.width, height=self.height)
            self.viewer.setBackgroundColor('white')
            self.viewer.removeAllModels()  # Clear any existing models
        except Exception as e:
            logger.error(f"Failed to initialize viewer: {e}")
            raise VisualizationError(f"Failed to initialize viewer: {e}")

    def process_pdb(self, pdb_path: str, metadata: Optional[dict] = None) -> Dict:
        """Process a single PDB file and generate visualizations."""
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        results = {
            'pdb_id': pdb_path.stem,
            'metadata': metadata or {},
            'visualizations': []
        }

        try:
            with open(pdb_path, 'r') as f:
                pdb_content = f.read()

            for rep_type in self.REPRESENTATION_TYPES:
                for orientation in self.ORIENTATIONS:
                    viz_result = self._render_view(
                        pdb_content,
                        results['pdb_id'],
                        rep_type,
                        orientation
                    )
                    if viz_result:
                        results['visualizations'].append(viz_result)

            return results

        except Exception as e:
            logger.error(f"Failed to process {pdb_path.name}: {e}")
            raise VisualizationError(f"Failed to process PDB file: {e}")

    def _render_view(self, pdb_content: str, pdb_id: str, rep_type: str, orientation: str) -> Optional[Dict]:
        """Render a specific view of the protein."""
        try:
            # Reinitialize viewer for each view
            self._initialize_viewer()

            # Add model and apply style
            self.viewer.addModel(pdb_content, "pdb")
            self.viewer.setStyle({}, self.REPRESENTATION_TYPES[rep_type])

            # Set orientation
            rotation = self.ORIENTATIONS[orientation]['rotation']
            self.viewer.setView([0, 0, 0, *rotation])
            self.viewer.zoomTo()

            # Generate output filename and path
            filename = f"{pdb_id}_{rep_type}_{orientation}.png"
            output_path = self.output_dir / filename

            # Render and save
            png_data = self.viewer.png()
            png_bytes = base64.b64decode(png_data)

            with Image.open(BytesIO(png_bytes)) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output_path, 'PNG', optimize=True)

            return {
                'path': str(output_path),
                'style': rep_type,
                'orientation': orientation
            }

        except Exception as e:
            logger.error(f"Failed to render view {rep_type}-{orientation} for {pdb_id}: {e}")
            return None

    @staticmethod
    def get_center_of_mass(coords: np.ndarray, masses: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate center of mass of protein structure."""
        if masses is None:
            masses = np.ones(len(coords))

        total_mass = masses.sum()
        if total_mass == 0 or coords.size == 0:
            return np.zeros(3)

        return (coords * masses[:, np.newaxis]).sum(axis=0) / total_mass