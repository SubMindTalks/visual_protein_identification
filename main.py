import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
from multiprocessing import Pool
import logging
from config.settings import DATA_DIR, OUTPUT_DIR, MODEL
from src.utils import create_directory, fetch_protein_metadata
from src.visualizer import ProteinVisualizer
from src.dataset import DataManager
from src.classifier import ProteinClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Protein Structure Classification Pipeline')
    parser.add_argument('--mode', choices=['visualize', 'train', 'evaluate', 'full'],
                        default='full', help='Pipeline mode')
    parser.add_argument('--model-path', type=str, help='Path to saved model for evaluation')
    return parser.parse_args()

def setup_pipeline() -> dict:
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

def process_protein_file(args):
    """Process a single protein file."""
    pdb_path, output_dir = args
    try:
        # Extract UniProt ID from the filename
        protein_id = pdb_path.stem.split("-")[1]  # Assumes format 'AF-<UniProtID>-...'
        metadata = fetch_protein_metadata(protein_id)
        if metadata is None:
            raise ValueError(f"Metadata not found for protein ID: {protein_id}")
        visualizer = ProteinVisualizer(str(output_dir))
        result = visualizer.process_pdb(str(pdb_path), metadata)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Failed to process {pdb_path.name}: {e}")
        return {"success": False, "error": str(e), "file": str(pdb_path)}

def process_proteins(pdb_dir: Path, output_dir: Path) -> dict:
    """Process protein structures and generate visualizations."""
    pdb_files = list(pdb_dir.glob("*.pdb"))
    logger.info(f"Found {len(pdb_files)} PDB files to process")
    with Pool() as pool:
        results = pool.map(process_protein_file, [(pdb, output_dir) for pdb in pdb_files])
    successful = [r['result'] for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    processing_summary = {
        "total_proteins": len(pdb_files),
        "successful": len(successful),
        "failed": len(failed),
        "failed_list": failed,
        "results": successful,
    }
    summary_path = output_dir / 'processing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(processing_summary, f, indent=2)
    logger.info(f"Processing complete. Success: {len(successful)}, Failed: {len(failed)}")
    return processing_summary

def prepare_dataset(processing_results: dict, data_dir: Path) -> DataManager:
    """Prepare dataset from processing results."""
    data_manager = DataManager(str(data_dir))
    label_mapping = {}
    for result in processing_results['results']:
        protein_id = result['pdb_id']
        if 'metadata' in result and 'protein_class' in result['metadata']:
            label_mapping[protein_id] = result['metadata']['protein_class']
    logger.info(f"Found {len(label_mapping)} proteins with valid labels")
    data_manager.organize_data(processing_results['results'], label_mapping)
    return data_manager

def train_model(data_manager: DataManager, model_dir: Path) -> ProteinClassifier:
    """Train the classification model."""
    dataloaders = data_manager.create_dataloaders(batch_size=MODEL['batch_size'])
    logger.info(f"Created dataloaders with batch size {MODEL['batch_size']}")
    num_classes = len(set(data_manager.label_mapping.values()))
    classifier = ProteinClassifier(num_classes=num_classes)
    logger.info(f"Initialized classifier for {num_classes} classes")
    logger.info("Training model")
    history = classifier.train(
        dataloaders['train'],
        dataloaders['val'],
        num_epochs=MODEL['num_epochs'],
        save_dir=str(model_dir)
    )
    history_path = model_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info("Model training complete")
    return classifier

def evaluate_model(classifier: ProteinClassifier, data_manager: DataManager,
                   results_dir: Path) -> None:
    """Evaluate the trained model."""
    dataloaders = data_manager.create_dataloaders(batch_size=MODEL['batch_size'])
    class_names = sorted(set(data_manager.label_mapping.values()))
    results = classifier.analyze_results(
        dataloaders['test'],
        class_names=class_names,
        output_dir=str(results_dir)
    )
    logger.info("Model evaluation complete")
    logger.info(f"Test accuracy: {results['classification_report']['accuracy']:.4f}")


def main():
    """Main pipeline execution."""
    args = parse_arguments()
    logger.info("Starting protein structure classification pipeline")
    logger.info(f"Pipeline mode: {args.mode}")
    try:
        directories = setup_pipeline()
        logger.info(f"Created pipeline directories at {directories['run']}")

        if args.mode in ['visualize', 'full']:
            logger.info("=== Visualization Phase ===")
            processing_results = process_proteins(DATA_DIR, directories['images'])

        if args.mode in ['train', 'full']:
            logger.info("=== Training Phase ===")
            data_manager = prepare_dataset(processing_results, directories['run'])
            classifier = train_model(data_manager, directories['models'])

        if args.mode in ['evaluate', 'full']:
            logger.info("=== Evaluation Phase ===")
            if args.mode == 'full':
                # Ensure `data_manager` is prepared if full pipeline is running
                data_manager = prepare_dataset(processing_results, directories['run'])
            if args.model_path:
                logger.info(f"Loading model from {args.model_path}")
                classifier = ProteinClassifier(num_classes=len(data_manager.label_mapping.values()))
                classifier.load_model(args.model_path)
            else:
                logger.error("Model path must be provided for evaluation.")
                raise ValueError("No model path provided for evaluation.")

            evaluate_model(classifier, data_manager, directories['results'])

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


