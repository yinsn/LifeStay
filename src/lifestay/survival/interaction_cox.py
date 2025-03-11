"""
Cox Proportional Hazards Model with Interactions Module

This module provides functionality for creating and evaluating Cox Proportional
Hazards models with interaction terms between a focal feature and other features.
"""

from typing import List, Optional

import pandas as pd

from .cox_model import CoxPHModel


def cox_model_with_interactions(
    X: pd.DataFrame,
    T: pd.Series,
    E: pd.Series,
    focal_feature: str,
    interaction_features: Optional[List[str]] = None,
    penalizer: float = 0.1,
) -> CoxPHModel:
    """
    Fit a Cox Proportional Hazards model with interaction terms between a focal feature
    and other selected features.

    Args:
        X (pd.DataFrame): Feature matrix
        T (pd.Series): Survival times
        E (pd.Series): Event indicators
        focal_feature (str): The feature to interact with other features
        interaction_features (List[str], optional): List of features to interact with the focal feature.
                                                   If None, interacts with all other features.
        penalizer (float, optional): Regularization strength. Defaults to 0.1.

    Returns:
        CoxPHModel: Fitted Cox model with interaction terms

    Example:
        ```python
        import pandas as pd
        from lifestay.survival import cox_model_with_interactions

        # Example with clinical data
        # Assume age is our focal feature and we want to see how it interacts with other features
        X = pd.DataFrame({
            'age': [65, 70, 55, 45, 60],
            'bmi': [24.5, 27.3, 22.1, 30.5, 25.8],
            'glucose': [90, 110, 85, 95, 105]
        })
        T = pd.Series([100, 80, 150, 120, 90])  # Survival times
        E = pd.Series([1, 1, 0, 0, 1])  # Event indicators (1=event occurred, 0=censored)

        # Fit model with interactions between age and all other features
        model = cox_model_with_interactions(X, T, E, focal_feature='age')
        ```
    """
    print(f"\n--- Cox Model with Interactions (Focal Feature: {focal_feature}) ---")

    # Validate that the focal feature exists in the dataframe
    if focal_feature not in X.columns:
        raise ValueError(f"Focal feature '{focal_feature}' not found in dataframe")

    # If no specific interaction features provided, use all features except the focal one
    if interaction_features is None:
        interaction_features = [col for col in X.columns if col != focal_feature]
    else:
        # Validate that all interaction features exist in the dataframe
        missing_features = [f for f in interaction_features if f not in X.columns]
        if missing_features:
            raise ValueError(
                f"Interaction features {missing_features} not found in dataframe"
            )

        # Remove focal feature from interaction_features if it's included
        if focal_feature in interaction_features:
            interaction_features.remove(focal_feature)
            print(
                f"Removed focal feature '{focal_feature}' from interaction features list"
            )

    # Create a copy of the original dataframe for adding interaction terms
    X_with_interactions = X.copy()

    # Generate interaction terms
    print(
        f"Creating {len(interaction_features)} interaction terms with '{focal_feature}'"
    )

    for feature in interaction_features:
        # Create interaction term column name
        interaction_name = f"{focal_feature}:{feature}"

        # Create interaction term
        # In statistical terms, an interaction means that the effect of one variable
        # depends on the value of another variable. In Cox models, this allows for
        # modeling how the hazard ratio of one feature changes based on the value
        # of another feature.
        #
        # Example: If age:bmi is positive, it means the effect of BMI on hazard
        # increases with age (or equivalently, the effect of age increases with BMI).
        X_with_interactions[interaction_name] = X[focal_feature] * X[feature]

    # Create and fit the model with regularization to prevent overfitting
    model = CoxPHModel(penalizer=penalizer)

    # Prepare data for lifelines
    lifelines_df = X_with_interactions.copy()
    lifelines_df["duration"] = T
    lifelines_df["event"] = E

    # Fit the model
    model.model.fit(df=lifelines_df, duration_col="duration", event_col="event")

    # Store data in model for convenience
    model.X = X_with_interactions
    model.T = T
    model.E = E
    model.focal_feature = focal_feature
    model.interaction_features = interaction_features
    model.fitted = True

    # Evaluate model
    evaluation = model.evaluate()
    print(f"\nConcordance Index: {evaluation['concordance_index']:.4f}")

    # Print information about interaction terms
    interaction_cols = [col for col in X_with_interactions.columns if ":" in col]
    if interaction_cols:
        print("\nInteraction Terms:")
        summary_df = model.summary()

        try:
            # Try to extract the interaction terms from the model summary
            interaction_summary = summary_df.loc[interaction_cols]
            significant_interactions = interaction_summary[
                interaction_summary["p"] < 0.05
            ]

            if not significant_interactions.empty:
                print(
                    f"Significant interactions (p < 0.05): {len(significant_interactions)}/{len(interaction_cols)}"
                )
                for idx, row in significant_interactions.iterrows():
                    print(f"  {idx}: coef={row['coef']:.4f}, p={row['p']:.4f}")
            else:
                print("No statistically significant interactions found (p < 0.05)")
        except Exception as e:
            print(f"Could not extract detailed interaction statistics: {e}")

    return model
