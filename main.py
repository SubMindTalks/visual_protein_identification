import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.settings import DATA_DIR, OUTPUT_DIR
from src.utils import create_directory, fetch_protein_metadata
from src.visualizer import ProteinVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_protein_file(pdb_path: Path, output_dir: Path) -> dict:
    """Process a single protein file with proper viewer initialization."""
    try:
        protein_id = pdb_path.stem.split("-")[1]
        metadata = fetch_protein_metadata(protein_id)

        # Create new visualizer instance for each protein
        visualizer = ProteinVisualizer(str(output_dir))
        result = visualizer.process_pdb(str(pdb_path), metadata)

        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Failed to process {pdb_path.name}: {str(e)}")
        return {"success": False, "error": str(e), "file": str(pdb_path)}


def process_proteins(pdb_dir: Path, output_dir: Path, max_workers: int = 4) -> dict:
    """Process protein structures with thread pooling."""
    pdb_files = list(pdb_dir.glob("*.pdb"))
    logger.info(f"Found {len(pdb_files)} PDB files to process")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdb = {
            executor.submit(process_protein_file, pdb, output_dir): pdb
            for pdb in pdb_files
        }

        for future in as_completed(future_to_pdb):
            pdb = future_to_pdb[future]
            try:
                result = future.result()
                results.append(result)
                if result["success"]:
                    logger.info(f"Successfully processed {pdb.name}")
                else:
                    logger.error(f"Failed to process {pdb.name}: {result['error']}")
            except Exception as e:
                logger.error(f"Error processing {pdb.name}: {str(e)}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "file": str(pdb)
                })

    successful = [r["result"] for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    summary = {
        "total_proteins": len(pdb_files),
        "successful": len(successful),
        "failed": len(failed),
        "failed_list": failed,
        "results": successful
    }

    summary_path = output_dir / "processing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    """Main execution with proper directory setup."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUT_DIR / f"run_{timestamp}"
        image_dir = run_dir / "images"

        for directory in [run_dir, image_dir]:
            create_directory(directory)

        logger.info(f"Created output directories at {run_dir}")

        summary = process_proteins(DATA_DIR, image_dir)
        logger.info(f"Processing complete. Success: {summary['successful']}, Failed: {summary['failed']}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()