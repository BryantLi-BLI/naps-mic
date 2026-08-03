#!/usr/bin/env python
"""
Batch processor for Onsager transport analysis.

Features:
- Configurable parallelism (default: 6 workers)
- Resume capability (skips already-processed files)
- Progress tracking with tqdm
- Logging to file and console
- Memory management with explicit garbage collection

Usage:
    python run_batch.py --n_jobs 6 --resume
    nohup python run_batch.py --n_jobs 6 --resume > batch.log 2>&1 &
"""

import os
import gc
import sys
import argparse
import logging
from glob import glob
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from onsager_io import get_processed_indices
from onsager_processing import process_single_xdatcar


def setup_logging(log_file):
    """Configure logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Batch process XDATCAR files for Onsager analysis')
    parser.add_argument('--input_dir', type=str,
                        default='/Users/bli/globus_dir/17.NaPS_al/03.RDF_diffusivity_data_benchmarks/amorphous_diff_data_new_pt3_10ns',
                        help='Directory containing XDATCAR files')
    parser.add_argument('--output_dir', type=str,
                        default='./Onsager_analyzers',
                        help='Output directory for results')
    parser.add_argument('--n_jobs', type=int, default=6,
                        help='Number of parallel workers (default: 6)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous run, skipping completed files')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Process files in batches of this size for memory management')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: only process first 5 files')
    return parser.parse_args()


def process_batch(paths_with_indices, figures_dir, output_dir, n_jobs, logger):
    """
    Process a batch of files in parallel.
    Returns list of results (dicts or None for failures).
    """
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_single_xdatcar)(path, idx, figures_dir, output_dir)
        for idx, path in paths_with_indices
    )

    # Force garbage collection after each batch
    gc.collect()

    return results


def main():
    args = parse_args()

    # Setup directories
    output_dir = args.output_dir
    figures_dir = os.path.join(output_dir, 'msd_figures')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'batch_processing_{timestamp}.log')
    logger = setup_logging(log_file)

    logger.info('=' * 60)
    logger.info('ONSAGER BATCH PROCESSING')
    logger.info('=' * 60)
    logger.info(f'Input directory: {args.input_dir}')
    logger.info(f'Output directory: {output_dir}')
    logger.info(f'Parallel workers: {args.n_jobs}')
    logger.info(f'Batch size: {args.batch_size}')
    logger.info(f'Resume mode: {args.resume}')
    logger.info(f'Test mode: {args.test}')

    # Discover XDATCAR files
    pattern = os.path.join(args.input_dir, '*')
    all_paths = sorted(glob(pattern))
    logger.info(f'Found {len(all_paths)} total files')

    # Build list of (index, path) tuples
    all_files = [(idx, path) for idx, path in enumerate(all_paths)]

    # Handle resume mode
    if args.resume:
        processed_indices = get_processed_indices(output_dir)
        logger.info(f'Already processed: {len(processed_indices)} files')
        files_to_process = [(idx, path) for idx, path in all_files if idx not in processed_indices]
        logger.info(f'Remaining to process: {len(files_to_process)} files')
    else:
        files_to_process = all_files

    # Test mode: only first 5 files
    if args.test:
        files_to_process = files_to_process[:5]
        logger.info(f'TEST MODE: Processing only {len(files_to_process)} files')

    if len(files_to_process) == 0:
        logger.info('No files to process. Exiting.')
        return

    # Process in batches
    all_results = []
    total_batches = (len(files_to_process) + args.batch_size - 1) // args.batch_size

    logger.info(f'Processing {len(files_to_process)} files in {total_batches} batches...')

    for batch_num in range(total_batches):
        start_idx = batch_num * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(files_to_process))
        batch = files_to_process[start_idx:end_idx]

        logger.info(f'Batch {batch_num + 1}/{total_batches}: Processing files {start_idx + 1} to {end_idx}')

        # Process batch with progress bar
        results = []
        with tqdm(total=len(batch), desc=f'Batch {batch_num + 1}', unit='file') as pbar:
            batch_results = Parallel(n_jobs=args.n_jobs, verbose=0)(
                delayed(process_single_xdatcar)(path, idx, figures_dir, output_dir)
                for idx, path in batch
            )
            results.extend(batch_results)
            pbar.update(len(batch))

        # Count successes/failures
        successes = sum(1 for r in results if r is not None)
        failures = sum(1 for r in results if r is None)
        logger.info(f'Batch {batch_num + 1} complete: {successes} successes, {failures} failures')

        all_results.extend(results)

        # Force garbage collection between batches
        gc.collect()
        logger.info('Garbage collection complete')

    # Filter out None results (failed processing)
    valid_results = [r for r in all_results if r is not None]

    logger.info('=' * 60)
    logger.info('PROCESSING COMPLETE')
    logger.info(f'Total processed: {len(valid_results)}/{len(files_to_process)} files')
    logger.info('=' * 60)

    # Save summary CSV
    if valid_results:
        df = pd.DataFrame(valid_results)
        csv_path = os.path.join(output_dir, f'onsager_results_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f'Results saved to: {csv_path}')

    logger.info('Done!')


if __name__ == '__main__':
    main()
