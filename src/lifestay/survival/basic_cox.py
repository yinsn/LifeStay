"""
Basic Cox Proportional Hazards Model Module

This module provides a simplified interface for creating and evaluating
basic Cox Proportional Hazards models for survival analysis.
"""

import pandas as pd

from .cox_model import CoxPHModel


def basic_cox_model(X: pd.DataFrame, T: pd.Series, E: pd.Series) -> CoxPHModel:
    """
    Fit a basic Cox Proportional Hazards model and demonstrate core functionality.

    Args:
        X (pd.DataFrame): Feature matrix
        T (pd.Series): Survival times
        E (pd.Series): Event indicators

    Returns:
        CoxPHModel: Fitted Cox model
    """
    print("\n--- Basic Cox Model ---")

    # Create and fit model
    model = CoxPHModel(
        penalizer=0.1, focal_feature=None
    )  # Some regularization to prevent overfitting

    # Prepare data for lifelines
    lifelines_df = X.copy()
    lifelines_df["duration"] = T
    lifelines_df["event"] = E

    # Fit the model
    model.model.fit(df=lifelines_df, duration_col="duration", event_col="event")

    # Store data in model for convenience
    model.X = X
    model.T = T
    model.E = E
    model.fitted = True

    # Evaluate model
    evaluation = model.evaluate()
    print(f"\nConcordance Index: {evaluation['concordance_index']:.4f}")

    return model
