"""
Metrics Calculation Module for Survival Analysis

This module provides functions for calculating various metrics related to
survival analysis models, including risk scores and feature contributions.
"""

from typing import Optional

import pandas as pd
from lifelines.fitters.coxph_fitter import CoxPHFitter as CoxPHModel

from lifestay.survival.contribution_analysis import calculate_feature_contribution


def calculate_risk_metrics(
    model: CoxPHModel,
    focal_feature: Optional[str] = None,
    interaction_pattern: str = ":",
) -> pd.DataFrame:
    """
    Calculate risk scores and feature contributions from the model for the same samples.

    This function calculates:
    1. Standard risk scores from the Cox model
    2. Feature contribution scores for the focal feature

    Both metrics are calculated for the same samples and stored in a DataFrame
    to ensure proper correspondence.

    Note:
        This function was extracted from the RiskComparisonVisualizer._calculate_metrics
        method to provide standalone access to this functionality.

    Args:
        model: The fitted Cox proportional hazards model
        focal_feature: The feature to calculate contributions for
        interaction_pattern: The pattern used to identify interaction terms in the model

    Returns:
        DataFrame containing risk scores, contributions, event status, and time for each sample

    Raises:
        ValueError: If model is None or focal feature cannot be determined
    """
    if model is None:
        raise ValueError("Model must be provided to calculate metrics")

    # Get data from the model
    X = model.X
    T = model.T
    E = model.E

    # Calculate standard risk scores
    risk_scores = model.predict_risk(X)

    # Calculate feature contributions
    if focal_feature is None:
        # Try to get focal feature from model metadata
        if hasattr(model, "focal_feature"):
            focal_feature = model.focal_feature
        else:
            raise ValueError(
                "Focal feature must be provided for contribution calculation"
            )

    contributions = calculate_feature_contribution(
        model=model,
        X=X,
        focal_feature=focal_feature,
        interaction_pattern=interaction_pattern,
    )

    # Store both metrics in a DataFrame to ensure they correspond to the same samples
    analysis_df = pd.DataFrame(
        {
            "risk_score": risk_scores,
            "contribution": contributions,
            "event": E.astype(int),
            "time": T,
        }
    )

    return analysis_df
