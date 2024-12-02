import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
from scipy.spatial.transform import Rotation
import seaborn as sns


class EnhancedProteinVisualizer:
    def __init__(self, pdb_file):
        self.parser = PDBParser(QUIET=True)
        self.structure = self.parser.get_structure("protein", pdb_file)
        self.setup_amino_acid_properties()
        self.atoms, self.residues = self.extract_structure_info()

    def setup_amino_acid_properties(self):
        """Define amino acid properties and color schemes"""
        # Define amino acid groups
        self.aa_groups = {
            'nonpolar': ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PRO', 'PHE', 'TRP'],
            'polar': ['SER', 'THR', 'CYS', 'TYR', 'ASN', 'GLN'],
            'acidic': ['ASP', 'GLU'],
            'basic': ['LYS', 'ARG', 'HIS']
        }

        # Create color palettes for each group
        nonpolar_palette = sns.color_palette("Reds", n_colors=len(self.aa_groups['nonpolar']))
        polar_palette = sns.color_palette("Blues", n_colors=len(self.aa_groups['polar']))
        acidic_palette = sns.color_palette("Greens", n_colors=len(self.aa_groups['acidic']))
        basic_palette = sns.color_palette("Purples", n_colors=len(self.aa_groups['basic']))

        # Create color mapping for individual amino acids
        self.aa_colors = {}

        # Assign colors to each amino acid within their group
        for aa, color in zip(self.aa_groups['nonpolar'], nonpolar_palette):
            self.aa_colors[aa] = color
        for aa, color in zip(self.aa_groups['polar'], polar_palette):
            self.aa_colors[aa] = color
        for aa, color in zip(self.aa_groups['acidic'], acidic_palette):
            self.aa_colors[aa] = color
        for aa, color in zip(self.aa_groups['basic'], basic_palette):
            self.aa_colors[aa] = color

        # Create group color mapping for the categorical representation
        self.group_colors = {
            'nonpolar': '#FFB6C1',  # Light red
            'polar': '#ADD8E6',  # Light blue
            'acidic': '#90EE90',  # Light green
            'basic': '#DDA0DD'  # Light purple
        }

    def get_aa_group(self, residue_name):
        """Determine which group an amino acid belongs to"""
        for group, residues in self.aa_groups.items():
            if residue_name in residues:
                return group
        return None

    def extract_structure_info(self):
        """Extract atomic coordinates and residue information"""
        atoms = []
        residues = []

        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in sum(self.aa_groups.values(), []):
                        for atom in residue:
                            atoms.append(atom.coord)
                            residues.append({
                                'name': residue.get_resname(),
                                'group': self.get_aa_group(residue.get_resname()),
                                'atom_name': atom.get_name()
                            })

        return np.array(atoms), residues

    def rotate_structure(self, angles):
        """Rotate atomic coordinates"""
        rot = Rotation.from_euler('xyz', angles, degrees=True)
        return rot.apply(self.atoms)

    def create_visualization(self, view_type='amino_acid'):
        """Create visualization with multiple rotations"""
        angles = np.arange(0, 360, 45)
        n_rows = 3
        n_cols = len(angles) // 3

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
        fig.suptitle(f'Protein Structure Projections - {view_type.replace("_", " ").title()} View',
                     fontsize=16, y=0.95)

        # Create color legend data
        if view_type == 'amino_acid':
            legend_colors = self.aa_colors
            legend_title = 'Amino Acids'
        else:  # group view
            legend_colors = self.group_colors
            legend_title = 'Chemical Groups'

        for i, axis in enumerate(['X', 'Y', 'Z']):
            for j, angle in enumerate(angles[:(n_cols)]):
                rotation = np.zeros(3)
                rotation[i] = angle

                rotated_coords = self.rotate_structure(rotation)
                projected_coords = rotated_coords[:, :2]

                ax = axes[i, j]

                # Plot points with appropriate coloring
                for coord, res in zip(projected_coords, self.residues):
                    color = (self.aa_colors[res['name']] if view_type == 'amino_acid'
                             else self.group_colors[res['group']])
                    ax.scatter(coord[0], coord[1], c=[color], alpha=0.7, s=30)

                ax.set_title(f'{axis}-axis rotation: {angle}°')
                ax.grid(True)
                ax.set_aspect('equal')

        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=color, label=name, markersize=10)
                           for name, color in legend_colors.items()]
        fig.legend(handles=legend_elements, title=legend_title,
                   loc='center right', bbox_to_anchor=(0.98, 0.5))

        plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        return fig


def main():
    # Replace with your PDB file path
    pdb_file = "example.pdb"
    visualizer = EnhancedProteinVisualizer(pdb_file)

    # Create both types of visualizations
    amino_acid_fig = visualizer.create_visualization(view_type='amino_acid')
    group_fig = visualizer.create_visualization(view_type='group')

    # Save the figures
    amino_acid_fig.savefig('protein_aa_projections.png', dpi=300, bbox_inches='tight')
    group_fig.savefig('protein_group_projections.png', dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()