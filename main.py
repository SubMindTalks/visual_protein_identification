"""Main script for protein structure classification pipeline."""
import sys
from pathlib import Path
import argparse
import json
from typing import Dict, Any
import torch
from datetime import datetime
from config.settings import (
    DATA_DIR, OUTPUT_DIR, MODEL_DIR,
    VISUALIZATION, DATASET, MODEL
)
from src.utils import create_directory, fetch_protein_metadata
from src.visualizer import ProteinVisualizer
from src.dataset import DataManager
from src.classifier import ProteinClassifier


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Protein Structure Classification Pipeline')
    parser.add_argument('--mode', choices=['visualize', 'train', 'evaluate', 'full'],
                        default='full', help='Pipeline mode')
    parser.add_argument('--model-path', type=str, help='Path to saved model for evaluation')
    return parser.parse_args()


def setup_pipeline() -> Dict[str, Path]:
    """Set up pipeline directories and return paths."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = OUTPUT_DIR / f"run_{timestamp}"
    directories = {
        'run': run_dir,
        'images': run_dir / 'images',
        'models': run_dir / 'models',
        'results': run_dir / 'results',
    }
    for directory in directories.values():
        create_directory(directory)
    return directories


def process_proteins(pdb_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Process protein structures and generate visualizations."""
    visualizer = ProteinVisualizer(str(output_dir))
    results = []
    failed_proteins = []
    pdb_files = list(pdb_dir.glob("*.pdb"))
    total_files = len(pdb_files)
    print(f"Found {total_files} PDB files to process")
    for idx, pdb_path in enumerate(pdb_files, 1):
        try:
            print(f"Processing {pdb_path.name} ({idx}/{total_files})")
            protein_id = pdb_path.stem
            metadata = fetch_protein_metadata(protein_id)
            result = visualizer.process_pdb(str(pdb_path), metadata)
            results.append(result)
            print(f"Successfully processed {protein_id}")
            if idx % 10 == 0:
                print(f"Processed {idx}/{total_files} proteins")
        except Exception as e:
            print(f"Failed to process {pdb_path.name}: {e}")
            failed_proteins.append(str(pdb_path))
    processing_summary = {
        'total_proteins': total_files,
        'successful': len(results),
        'failed': len(failed_proteins),
        'failed_list': failed_proteins,
        'results': results
    }
    summary_path = output_dir / 'processing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(processing_summary, f, indent=2)
    print(f"Processing complete. Success: {len(results)}, Failed: {len(failed_proteins)}")
    return processing_summary


def prepare_dataset(processing_results: Dict[str, Any], data_dir: Path) -> DataManager:
    """Prepare dataset from processing results."""
    data_manager = DataManager(str(data_dir))
    label_mapping = {}
    for result in processing_results['results']:
        protein_id = result['pdb_id']
        if 'metadata' in result and 'protein_class' in result['metadata']:
            label_mapping[protein_id] = result['metadata']['protein_class']
    print(f"Found {len(label_mapping)} proteins with valid labels")
    data_manager.organize_data(processing_results['results'], label_mapping)
    return data_manager


def train_model(data_manager: DataManager, model_dir: Path) -> ProteinClassifier:
    """Train the classification model."""
    dataloaders = data_manager.create_dataloaders(
        batch_size=MODEL['batch_size']
    )
    print(f"Created dataloaders with batch size {MODEL['batch_size']}")
    num_classes = len(set(label_mapping.values()))
    classifier = ProteinClassifier(num_classes=num_classes)
    print(f"Initialized classifier for {num_classes} classes")
    print("Training model")
    history = classifier.train(
        dataloaders['train'],
        dataloaders['val'],
        num_epochs=MODEL['num_epochs'],
        save_dir=str(model_dir)
    )
    history_path = model_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print("Model training complete")
    return classifier


def evaluate_model(classifier: ProteinClassifier, data_manager: DataManager,
                   results_dir: Path) -> None:
    """Evaluate the trained model."""
    dataloaders = data_manager.create_dataloaders(batch_size=MODEL['batch_size'])
    class_names = sorted(set(label_mapping.values()))
    results = classifier.analyze_results(
        dataloaders['test'],
        class_names=class_names,
        output_dir=str(results_dir)
    )
    print("Model evaluation complete")
    print(f"Test accuracy: {results['classification_report']['accuracy']:.4f}")


def main():
    """Main pipeline execution."""
    args = parse_arguments()
    print("Starting protein structure classification pipeline")
    print(f"Pipeline mode: {args.mode}")
    try:
        directories = setup_pipeline()
        print(f"Created pipeline directories at {directories['run']}")
        if args.mode in ['visualize', 'full']:
            print("=== Visualization Phase ===")
            processing_results = process_proteins(DATA_DIR, directories['images'])
        if args.mode in ['train', 'full']:
            print("=== Training Phase ===")
            data_manager = prepare_dataset(processing_results, directories['run'])
            classifier = train_model(data_manager, directories['models'])
        if args.mode in ['evaluate', 'full']:
            print("=== Evaluation Phase ===")
            if args.model_path:
                classifier.load_model(args.model_path)
            evaluate_model(classifier, data_manager, directories['results'])
        print("Pipeline completed successfully")
    except Exception as e:
        print(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()