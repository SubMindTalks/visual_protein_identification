"""Protein visualization module using Py3Dmol."""

import logging
import py3Dmol
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from .utils import create_directory

logger = logging.getLogger(__name__)

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
        'cartoon': {'cartoon': {}},
        'ribbon': {'cartoon': {'style': 'ribbon'}}
    }

    def __init__(self, output_dir: str, width: int = 800, height: int = 800):
        self.output_dir = create_directory(output_dir)
        self.width = width
        self.height = height
        self._setup_viewer()

    def _setup_viewer(self):
        """Initialize the Py3Dmol viewer."""
        try:
            self.viewer = py3Dmol.view(width=self.width, height=self.height, viewergrid=(1, 1))
            self.viewer.setBackgroundColor('white')
            logger.info("Viewer successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize viewer: {e}")
            self.viewer = None  # Explicitly set to None if initialization fails

    def process_pdb(self, pdb_path: str, metadata: Optional[dict] = None) -> Dict:
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

            # Reset viewer for each PDB
            self._setup_viewer()

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
            raise

    def _render_view(self, pdb_content: str, pdb_id: str, rep_type: str, orientation: str) -> Optional[Dict]:
        if not self.viewer:
            logger.error("Viewer must be initialized before generating images.")
            return None

        try:
            # Clear previous models
            self.viewer.removeAllModels()

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
            if not png_data:
                logger.error(f"Failed to generate PNG data for {rep_type}-{orientation} view.")
                return None

            png_bytes = base64.b64decode(png_data)
            with Image.open(BytesIO(png_bytes)) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output_path, 'PNG')

            return {
                'path': str(output_path),
                'style': rep_type,
                'orientation': orientation
            }

        except Exception as e:
            logger.error(f"Failed to render view {rep_type}-{orientation} for {pdb_id}: {e}")
            return None
