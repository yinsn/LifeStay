"""
Cox Proportional Hazards Model

This module provides an implementation of the Cox Proportional Hazards model
for survival analysis, utilizing the lifelines library.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from .data_converter import convert_to_lifelines_format


class CoxPHModel:
    """
    Cox Proportional Hazards Model for survival analysis.

    This class wraps the lifelines CoxPHFitter and provides additional
    functionality for data preprocessing, model evaluation, and result interpretation.
    """

    def __init__(
        self,
        penalizer: float = 0.0,
        l1_ratio: float = 0.0,
        alpha: float = 0.05,
        focal_feature: Optional[str] = None,
        interaction_features: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the Cox Proportional Hazards model.

        Args:
            penalizer: Coefficient penalization strength (L2 regularization)
            l1_ratio: L1 ratio when doing elastic net regularization (0=L2, 1=L1)
            alpha: Significance level for confidence intervals
            focal_feature: The feature to interact with other features (for interaction models)
            interaction_features: Features to interact with the focal feature
            **kwargs: Additional parameters to pass to lifelines CoxPHFitter
        """
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.focal_feature = focal_feature
        self.interaction_features = interaction_features

        # Additional attributes used by SurvivalDatasetBuilder
        self.heartbeat_column: Optional[str] = None
        self.window_size: int = 5  # Default window size
        self.sample_size: Optional[int] = None

        # Initialize the lifelines model
        self.model = CoxPHFitter(
            penalizer=penalizer, l1_ratio=l1_ratio, alpha=alpha, **kwargs
        )

        self.fitted = False

    def fit(
        self,
        data: pd.DataFrame,
        time_column: Optional[str] = None,
        event_column: str = "sample_type",
        event_value: str = "negative",
        feature_columns: Optional[List[str]] = None,
        time_from_window: bool = True,
        window_size_column: Optional[str] = None,
        **kwargs: Any,
    ) -> "CoxPHModel":
        """
        Fit the Cox Proportional Hazards model to the data.

        Args:
            data: DataFrame created by SampleBuilder
            time_column: Column to use as time-to-event
            event_column: Column that indicates the event type
            event_value: Value in event_column that represents the negative event
            feature_columns: List of columns to use as features
            time_from_window: Whether to generate time values based on window distance
            window_size_column: Column with window size information
            **kwargs: Additional parameters to pass to fit_dataframe

        Returns:
            Self (fitted model)
        """
        # Convert data to lifelines format
        X, T, E = convert_to_lifelines_format(
            data=data,
            time_column=time_column,
            event_column=event_column,
            event_value=event_value,
            feature_columns=feature_columns,
            time_from_window=time_from_window,
            window_size_column=window_size_column,
        )

        # Store the data for later use
        self.X = X
        self.T = T
        self.E = E

        # Create a DataFrame for lifelines
        lifelines_df = X.copy()
        lifelines_df["duration"] = T
        lifelines_df["event"] = E

        # Fit the model
        self.model.fit(
            df=lifelines_df, duration_col="duration", event_col="event", **kwargs
        )

        self.fitted = True
        return self

    def summary(self, **kwargs: Any) -> pd.DataFrame:
        """
        Get a summary of the fitted model.

        Args:
            **kwargs: Additional parameters to pass to lifelines summary

        Returns:
            DataFrame with model coefficients and statistics
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")

        # Handle both older lifelines versions (where summary is a method)
        # and newer versions (where summary is a DataFrame property)
        if callable(self.model.summary):
            return self.model.summary(**kwargs)
        else:
            # In newer lifelines versions, summary is a DataFrame property
            return self.model.summary

    def print_summary(self, **kwargs: Any) -> None:
        """
        Print a summary of the fitted model.

        Args:
            **kwargs: Additional parameters to pass to lifelines print_summary
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")

        # Handle both older and newer lifelines versions
        if callable(self.model.print_summary):
            self.model.print_summary(**kwargs)
        else:
            # In newer lifelines versions, we can print the summary DataFrame
            print(self.model.summary)

    def predict_risk(self, X: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Predict the risk score for the given data.

        Args:
            X: DataFrame with feature columns. If None, uses the training data

        Returns:
            Series of risk scores
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")

        if X is None:
            X = self.X

        return self.model.predict_partial_hazard(X)

    def predict_survival_function(
        self,
        X: Optional[pd.DataFrame] = None,
        times: Optional[Union[List[float], np.ndarray]] = None,
    ) -> pd.DataFrame:
        """
        Predict the survival function for the given data.

        Args:
            X: DataFrame with feature columns. If None, uses the training data
            times: Times to predict survival probability at. If None, uses default times

        Returns:
            DataFrame of survival probabilities
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")

        if X is None:
            X = self.X

        return self.model.predict_survival_function(X, times=times)

    def evaluate(
        self,
        X: Optional[pd.DataFrame] = None,
        T: Optional[pd.Series] = None,
        E: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the model performance using concordance index.

        Args:
            X: Feature DataFrame. If None, uses training data
            T: Time/duration Series. If None, uses training data
            E: Event Series. If None, uses training data

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")

        # Use training data if not provided
        if X is None:
            X = self.X
        if T is None:
            T = self.T
        if E is None:
            E = self.E

        # Predict risk scores
        risk_scores = self.predict_risk(X)

        # Calculate concordance index
        c_index = concordance_index(T, -risk_scores, E)

        return {"concordance_index": c_index}

    def check_assumptions(self, **kwargs: Any) -> None:
        """
        Check the proportional hazards assumption using lifelines diagnostics.

        Args:
            **kwargs: Additional parameters to pass to check_assumptions
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")

        # Prepare data in the format expected by lifelines
        X_check = self.X.copy()
        X_check["duration"] = self.T
        X_check["event"] = self.E

        try:
            # Check proportional hazards assumption
            # Try the approach for newer lifelines versions
            self.model.check_assumptions(X_check, p_value_threshold=0.05, **kwargs)
        except (TypeError, ValueError) as e:
            # Fall back to the approach for older lifelines versions or custom approach
            print(f"Could not check assumptions with standard method: {e}")
            print("Consider manual checking of proportional hazards assumption.")

    def plot_partial_effects(self, columns: List[str], **kwargs: Any) -> None:
        """
        Plot the partial effects of the specified columns.

        Args:
            columns: Columns to plot partial effects for
            **kwargs: Additional parameters to pass to plot_partial_effects_on_outcome
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")

        for column in columns:
            if column not in self.X.columns:
                raise ValueError(f"Column '{column}' not found in feature data")

        self.model.plot_partial_effects_on_outcome(covariates=columns, **kwargs)

    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.

        Args:
            filepath: Path to save the model to
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")

        self.model.save(filepath)

    @staticmethod
    def load_model(filepath: str) -> "CoxPHModel":
        """
        Load a model from a file.

        Args:
            filepath: Path to load the model from

        Returns:
            Loaded CoxPHModel instance
        """
        model = CoxPHModel()
        model.model = CoxPHFitter.load(filepath)
        model.fitted = True
        return model
