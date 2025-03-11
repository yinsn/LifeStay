"""
Test script for debugging the survival analysis module.

This script helps diagnose issues with the SampleBuilder and survival analysis workflow.
"""

import os
import sys
from typing import Dict, List, Optional

import pandas as pd

from lifestay.survival import (
    CoxPHModel,
    create_survival_dataset_from_sample_builder,
    fit_and_evaluate_cox_model,
)


def test_survival_analysis(
    file_path: str,
    target_columns: List[str],
    heartbeat_column: str,
    patterns: Dict[str, List[int]],
    window_size: int = 5,
    sample_size: int = 10000,
) -> Optional[CoxPHModel]:
    """
    Test the survival analysis workflow with debugging enabled.

    Args:
        file_path: Path to the CSV file
        target_columns: List of columns to use for analysis
        heartbeat_column: Column for EOL pattern detection
        patterns: Dict mapping pattern names to pattern definitions
        window_size: Number of values in each window
        sample_size: Number of samples to use for testing

    Returns:
        The fitted Cox model if successful, None otherwise
    """
    print(f"Starting survival analysis test with file: {file_path}")

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' does not exist!")
        return None

    # Try to read the first few rows of the CSV to get column names
    try:
        print("Reading CSV headers...")
        df_peek = pd.read_csv(file_path, nrows=5)
        print(f"CSV columns: {df_peek.columns.tolist()}")

        # Check if target columns exist
        missing_targets = [col for col in target_columns if col not in df_peek.columns]
        if missing_targets:
            print(
                f"WARNING: These target columns don't exist in the file: {missing_targets}"
            )

        # Check heartbeat column
        if heartbeat_column not in df_peek.columns:
            print(
                f"WARNING: Heartbeat column '{heartbeat_column}' not found in the file"
            )

    except Exception as e:
        print(f"Error reading CSV headers: {e}")
        return None

    # Try to create the survival dataset with debug info
    try:
        print("\nCreating survival dataset...")
        X, T, E = create_survival_dataset_from_sample_builder(
            file_path=file_path,
            target_columns=target_columns,
            heartbeat_column=heartbeat_column,
            window_size=window_size,
            patterns=patterns,
            sample_size=sample_size,
            debug=True,  # Enable debug mode
        )

        print(f"\nSurvival dataset created successfully!")
        print(f"Features shape: {X.shape}")
        print(f"Time values shape: {T.shape}")
        print(f"Event indicators shape: {E.shape}")
        print(f"Event ratio: {E.mean():.2f} ({E.sum()} events out of {len(E)} samples)")

        # Try to fit the model
        print("\nFitting Cox model...")
        model, train_metrics, test_metrics = fit_and_evaluate_cox_model(X, T, E)

        # Print results
        print("\nModel Summary:")
        model.print_summary()

        print("\nEvaluation Metrics:")
        print(f"Training Concordance Index: {train_metrics['concordance_index']:.4f}")
        print(f"Testing Concordance Index: {test_metrics['concordance_index']:.4f}")

        return model

    except Exception as e:
        print(f"Error in survival analysis: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_survival.py <csv_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    # Example configuration - modify these based on your actual data
    target_columns = ["col1", "col2", "col3"]  # Replace with actual column names
    heartbeat_column = "heartbeat"  # Replace with actual heartbeat column

    patterns = {
        "negative": [1, 0, 0],  # Pattern for negative samples (event occurred)
        "positive": [1, 1, 1],  # Pattern for positive samples (censored)
    }

    # Run test
    test_survival_analysis(
        file_path=file_path,
        target_columns=target_columns,
        heartbeat_column=heartbeat_column,
        patterns=patterns,
    )
