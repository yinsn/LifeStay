"""
Utility functions for survival analysis.

This module provides helper functions and examples for using the survival analysis modules.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .contribution_analysis import visualize_contribution_distribution
from .cox_model import CoxPHModel
from .data_builder import SurvivalDatasetBuilder


def create_survival_dataset_from_sample_builder(
    file_path: str,
    feature_columns: List[str],
    heartbeat_column: str,
    window_size: int,
    patterns: Dict[str, List[int]],
    time_column: Optional[str] = None,
    event_value: str = "negative",
    sample_size: Optional[int] = None,
    debug: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Create a survival dataset from a CSV file using SampleBuilder.

    Args:
        file_path: Path to the CSV file to process
        feature_columns: List of column names to calculate window averages for
                     and to use as features in the survival model
        heartbeat_column: Column name to use for EOL pattern detection
        window_size: Number of values in each average window
        patterns: Dict mapping pattern names to pattern definitions
        time_column: Column to use as time-to-event (if None, will use window index)
        event_value: Value in sample_type column that represents the event (will be converted to event=1)
        sample_size: Optional number of rows to process
        debug: Whether to print additional debugging information

    Returns:
        Tuple of (X, T, E) where:
        - X: DataFrame containing feature columns
        - T: Series containing duration/time values
        - E: Series containing event indicators
    """
    # Validate inputs
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not feature_columns:
        raise ValueError("Feature columns must be specified")

    if not heartbeat_column:
        raise ValueError("Heartbeat column must be specified")

    if not patterns:
        raise ValueError("Patterns must be specified")

    # Print debug information
    if debug:
        print(f"Creating dataset from file: {file_path}")
        print(f"Feature columns requested: {feature_columns}")
        print(f"Heartbeat column: {heartbeat_column}")
        print(f"Patterns: {patterns}")

    # Use SurvivalDatasetBuilder to build the dataset
    builder = SurvivalDatasetBuilder(
        feature_columns=feature_columns,
        heartbeat_column=heartbeat_column,
        patterns=patterns,
        window_size=window_size,
        event_value=event_value,
        time_column=time_column,
        debug=debug,
    )

    # Set the file path and sample size
    builder.with_file(file_path)
    if sample_size is not None:
        builder.with_sample_size(sample_size)

    # Make sure we use basic_cox_model
    builder.with_focal_feature(None)

    # Build the model and extract X, T, E
    model = builder.build()
    return model.X, model.T, model.E


def fit_and_evaluate_cox_model(
    X: pd.DataFrame,
    T: pd.Series,
    E: pd.Series,
    penalizer: float = 0.0,
    l1_ratio: float = 0.0,
    alpha: float = 0.05,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[CoxPHModel, Dict[str, float], Dict[str, float]]:
    """
    Fit a Cox proportional hazards model and evaluate on train and test sets.

    Args:
        X: Feature DataFrame
        T: Time/duration Series
        E: Event Series
        penalizer: Coefficient penalization strength
        l1_ratio: L1 ratio for elastic net regularization
        alpha: Significance level for confidence intervals
        test_fraction: Fraction of data to use for testing
        random_state: Random state for reproducibility

    Returns:
        Tuple of (fitted model, training metrics, test metrics)
    """
    # Split the data into train and test sets
    np.random.seed(random_state)
    n = len(X)
    test_size = int(n * test_fraction)
    test_indices = np.random.choice(n, test_size, replace=False)
    train_indices = np.array([i for i in range(n) if i not in test_indices])

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    T_train, T_test = T.iloc[train_indices], T.iloc[test_indices]
    E_train, E_test = E.iloc[train_indices], E.iloc[test_indices]

    # Create and fit the model
    model = CoxPHModel(penalizer=penalizer, l1_ratio=l1_ratio, alpha=alpha)

    # Create a DataFrame for lifelines
    lifelines_df = X_train.copy()
    lifelines_df["duration"] = T_train
    lifelines_df["event"] = E_train

    # Fit the model
    model.model.fit(df=lifelines_df, duration_col="duration", event_col="event")

    # Set the fitted data
    model.X = X_train
    model.T = T_train
    model.E = E_train
    model.fitted = True

    # Evaluate on train and test sets
    train_metrics = model.evaluate(X_train, T_train, E_train)
    test_metrics = model.evaluate(X_test, T_test, E_test)

    return model, train_metrics, test_metrics


def plot_survival_curves(
    model: CoxPHModel,
    X: Optional[pd.DataFrame] = None,
    num_samples: int = 5,
    random_state: int = 42,
) -> None:
    """
    Plot survival curves for a sample of observations.

    Args:
        model: Fitted CoxPHModel
        X: Feature DataFrame. If None, uses model's training data
        num_samples: Number of samples to plot
        random_state: Random state for reproducibility
    """
    if not model.fitted:
        raise ValueError("Model has not been fitted yet")

    if X is None:
        X = model.X

    # Select a random sample of observations
    np.random.seed(random_state)
    sample_indices = np.random.choice(len(X), num_samples, replace=False)
    X_sample = X.iloc[sample_indices]

    # Predict survival function
    sf = model.predict_survival_function(X_sample)

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Plot each survival curve
    for i, column in enumerate(sf.columns):
        plt.step(sf.index, sf[column], where="post", label=f"Sample {i + 1}", alpha=0.7)

    # Add median survival line
    plt.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Median Survival")

    # Add labels and legend
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Survival Curves for Selected Samples")
    plt.legend()
    plt.grid(alpha=0.3)

    # Show the plot
    plt.tight_layout()
    plt.show()


def example_usage() -> None:
    """
    Example function demonstrating how to use the survival analysis modules.
    """
    # Example parameters (these should be adjusted for your specific data)
    file_path = "your_data.csv"  # Path to your data file
    target_columns = ["feature1", "feature2", "feature3"]  # Columns to use as features
    heartbeat_column = "heartbeat"  # Column to use for EOL pattern detection
    window_size = 5  # Number of values in each window

    # Define patterns for positive and negative samples
    patterns = {
        "negative": [1, 0, 0],  # Pattern for negative samples (event occurred)
        "positive": [1, 1, 1],  # Pattern for positive samples (censored)
    }

    # Create survival dataset using the low-level function
    X, T, E = create_survival_dataset_from_sample_builder(
        file_path=file_path,
        feature_columns=target_columns,
        heartbeat_column=heartbeat_column,
        window_size=window_size,
        patterns=patterns,
        event_value="negative",  # Event value that represents the event occurred
    )

    # Print dataset summary
    print(f"Dataset created with {len(X)} samples")
    print(f"Features: {X.columns.tolist()}")
    print(f"Events: {E.sum()} occurred, {len(E) - E.sum()} censored")

    # Fit and evaluate model using the low-level function
    model, train_metrics, test_metrics = fit_and_evaluate_cox_model(X, T, E)

    # Print model summary
    print("\nModel Summary:")
    model.print_summary()

    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    print(f"Training Concordance Index: {train_metrics['concordance_index']:.4f}")
    print(f"Testing Concordance Index: {test_metrics['concordance_index']:.4f}")

    # Plot survival curves
    plot_survival_curves(model)

    # Check proportional hazards assumption
    print("\nChecking Proportional Hazards Assumption:")
    model.check_assumptions()

    # Example using SurvivalDatasetBuilder directly
    print("\n--- Using SurvivalDatasetBuilder ---")

    # Example 1: Basic Cox model
    builder = SurvivalDatasetBuilder(
        feature_columns=target_columns,
        heartbeat_column=heartbeat_column,
        patterns=patterns,
        window_size=window_size,
        event_value="negative",
    )

    builder.with_file(file_path)

    # For basic Cox model, either set focal_feature to None or don't set it (None is default)
    builder.with_focal_feature(None)

    # Build the model
    basic_model = builder.build()
    print(f"\nBasic Cox Model built with {len(basic_model.X)} samples")
    print(f"Concordance Index: {basic_model.evaluate()['concordance_index']:.4f}")

    # Example 2: Cox model with interactions
    builder = SurvivalDatasetBuilder(
        feature_columns=target_columns,
        heartbeat_column=heartbeat_column,
        patterns=patterns,
        window_size=window_size,
        event_value="negative",
    )

    builder.with_file(file_path)

    # For interaction Cox model, set a focal feature
    # Assuming 'feature1' is a valid feature in the dataset
    builder.with_focal_feature("feature1")

    # Build the model
    interaction_model = builder.build()
    print(f"\nInteraction Cox Model built with {len(interaction_model.X)} samples")
    print(f"Focal Feature: {interaction_model.focal_feature}")
    print(f"Concordance Index: {interaction_model.evaluate()['concordance_index']:.4f}")

    # Example of visualizing feature contributions
    contributions = visualize_contribution_distribution(interaction_model)


def create_dataset(
    file_path: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    heartbeat_column: Optional[str] = None,
    patterns: Optional[Dict[str, List[int]]] = None,
    window_size: int = 5,
    event_value: str = "negative",
    time_column: Optional[str] = None,
    sample_size: Optional[int] = None,
    debug: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Simplified function to create a survival dataset using the SurvivalDatasetBuilder.

    This function ensures metadata columns like user_id and eol_index are excluded from
    the feature matrix.

    Args:
        file_path: Path to the CSV file to process
        feature_columns: List of column names to calculate window averages for
                     and to use as features in the survival model
        heartbeat_column: Column name to use for EOL pattern detection
        patterns: Dict mapping pattern names to pattern definitions
        window_size: Number of values in each average window
        event_value: Value in sample_type column that represents the event (will be converted to event=1)
        time_column: Column to use as time-to-event (if None, will use window index)
        sample_size: Optional number of rows to process
        debug: Whether to print additional debugging information

    Returns:
        Tuple of (X, T, E) where:
        - X: DataFrame containing feature columns (excluding metadata like user_id and eol_index)
        - T: Series containing duration/time values
        - E: Series containing event indicators
    """
    # Set default patterns if not provided
    if patterns is None:
        patterns = {"negative": [1, 0, 0], "positive": [1, 1, 1]}

    # Create builder with parameters
    builder = SurvivalDatasetBuilder(
        feature_columns=feature_columns,
        heartbeat_column=heartbeat_column,
        patterns=patterns,
        window_size=window_size,
        event_value=event_value,
        time_column=time_column,
        debug=debug,
    )

    # Set file path and sample size if provided
    if file_path is not None:
        builder.with_file(file_path)

    if sample_size is not None:
        builder.with_sample_size(sample_size)

    # Explicitly set focal_feature to None to ensure we get basic_cox_model behavior
    builder.with_focal_feature(None)

    # Build the dataset but extract X, T, E from the model
    model = builder.build()
    return model.X, model.T, model.E
