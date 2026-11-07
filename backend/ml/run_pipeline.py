"""
Orchestration script: End-to-end AI layer setup and testing.
Runs: dataset_builder → trainer → evaluator → main.py with predictor.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_DIR = PROJECT_ROOT / 'backend' / 'ml'
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = ML_DIR
RESULTS_DIR = DATA_DIR

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(str(d), exist_ok=True)

AMQSS_CSV = RESULTS_DIR / 'market_scores.csv'
DATASET_CSV = DATA_DIR / 'dataset.csv'


def parse_args():
    parser = argparse.ArgumentParser(description='Run full AMQSS ML pipeline end-to-end')
    parser.add_argument('--price-csv', default=os.getenv('AMQSS_PRICE_CSV'), help='Path to raw price CSV')
    parser.add_argument('--results-dir', default=os.getenv('AMQSS_RESULTS_DIR', str(RESULTS_DIR)), help='Directory with market_scores.csv')
    parser.add_argument('--models-dir', default=os.getenv('AMQSS_MODELS_DIR', str(MODELS_DIR)), help='Directory for model artifacts')
    parser.add_argument('--data-dir', default=os.getenv('AMQSS_DATA_DIR', str(DATA_DIR)), help='Directory for dataset.csv')
    parser.add_argument('--test-fraction', type=float, default=0.2, help='Test fraction for trainer/evaluator')
    parser.add_argument('--n-estimators', type=int, default=100, help='Trees for trainer')
    return parser.parse_args()


def run_step(step_name, script_path, args=None):
    print(f"\n{step_name}")
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=str(PROJECT_ROOT))
        if result.returncode == 0:
            print(f"{step_name}: done")
            return True
        else:
            print(f"{step_name}: failed")
            return False
    except Exception as e:
        print(f"{step_name}: error: {e}")
        return False


def main():
    args = parse_args()

    data_dir = Path(args.data_dir).expanduser()
    models_dir = Path(args.models_dir).expanduser()
    results_dir = Path(args.results_dir).expanduser()

    for d in [data_dir, models_dir, results_dir]:
        os.makedirs(str(d), exist_ok=True)

    amqss_csv = results_dir / 'market_scores.csv'
    dataset_csv = data_dir / 'dataset.csv'
    legacy_default_csv = Path.home() / 'Downloads' / 'EURUSD_60_2025-01-20_2026-01-19.csv'
    fallback_data_csv = DATA_DIR / 'EURUSD_60_2025-01-20_2026-01-19.csv'

    if args.price_csv:
        price_csv = Path(args.price_csv).expanduser()
    elif legacy_default_csv.exists():
        price_csv = legacy_default_csv
    elif fallback_data_csv.exists():
        price_csv = fallback_data_csv
    else:
        print("\n✗ Price CSV is required.")
        print("  Use --price-csv <path> or set AMQSS_PRICE_CSV")
        return

    print("\nAMQSS AI layer setup")

    # Step 1: Build dataset
    print("\n[1/4] Building dataset")
    if not os.path.exists(str(amqss_csv)):
        print(f"No AMQSS results found at {amqss_csv}")
        print("    Run main.py first to generate market_scores.csv")
        print("    Skipping dataset builder for now.")
        dataset_ok = False
    else:
        dataset_builder_script = str(ML_DIR / 'dataset_builder.py')
        dataset_ok = run_step(
            "Dataset Builder",
            dataset_builder_script,
            args=[
                '--amqss-csv', str(amqss_csv),
                '--price-csv', str(price_csv),
                '--output-csv', str(dataset_csv)
            ]
        )

    if not dataset_ok:
        print("\nSkipping training without a dataset")
        return

    # Step 2: Train model
    print("\n[2/4] Training model")
    trainer_script = str(ML_DIR / 'trainer.py')
    trainer_ok = run_step(
        "Trainer",
        trainer_script,
        args=[
            '--dataset-csv', str(dataset_csv),
            '--models-dir', str(models_dir),
            '--test-fraction', str(args.test_fraction),
            '--n-estimators', str(args.n_estimators)
        ]
    )

    if not trainer_ok:
        print("\nTraining failed")
        return

    # Step 3: Evaluate
    print("\n[3/4] Evaluating model")
    evaluator_script = str(ML_DIR / 'evaluator.py')
    eval_ok = run_step(
        "Evaluator",
        evaluator_script,
        args=[
            '--dataset-csv', str(dataset_csv),
            '--models-dir', str(models_dir),
            '--test-fraction', str(args.test_fraction)
        ]
    )

    # Step 4: Verify inference with main.py
    print("\n[4/4] Verifying pipeline inference")
    main_script = str(PROJECT_ROOT / 'main.py')
    run_step(
        "Scoring Inference Test",
        main_script,
        args=[
            '--price-csv', str(price_csv),
            '--models-dir', str(models_dir),
            '--results-dir', str(results_dir)
        ]
    )

    print("\n✓ Pipeline execution complete.")


if __name__ == '__main__':
    main()
