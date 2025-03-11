"""
Feature Contribution Analysis for Survival Models

This module provides functionality for analyzing and visualizing the contribution
of features and their interaction terms to risk predictions in survival models.
"""

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

from .cox_model import CoxPHModel


def calculate_feature_contribution(
    model: CoxPHModel,
    X: Optional[pd.DataFrame] = None,
    focal_feature: Optional[str] = None,
    interaction_pattern: str = ":",
) -> pd.Series:
    """
    Calculate the net contribution of a focal feature and its interaction terms to the
    partial hazard in a Cox model.

    The contribution is calculated as:
        contribution_i = exp(β_focal * X_focal_i + sum(β_interaction_j * X_interaction_j_i))

    Where X_interaction_j_i are the interaction terms involving the focal feature.
    This returns the partial hazard form which ensures non-negativity.

    Args:
        model (CoxPHModel): Fitted Cox model with interaction terms
        X (pd.DataFrame, optional): Feature data. If None, uses training data from model.
        focal_feature (str, optional): Name of the focal feature to calculate contribution for.
                                      If None, uses the focal_feature attribute from the model.
                                      Can use the original column name without the "_avg" suffix.
        interaction_pattern (str, optional): Pattern used to identify interaction terms in the model.
                                           Defaults to ":".

    Returns:
        pd.Series: Net contribution of the focal feature and its interactions for each sample
                  in partial hazard form (exponential scale)
    """
    if not model.fitted:
        raise ValueError("Model has not been fitted yet")

    # Use training data if not provided
    if X is None:
        X = model.X

    # Use model's focal_feature if not provided
    if focal_feature is None:
        focal_feature = model.focal_feature

    # Determine the actual focal feature column name (with or without _avg suffix)
    focal_feature_col = focal_feature
    if focal_feature not in X.columns and f"{focal_feature}_avg" in X.columns:
        focal_feature_col = f"{focal_feature}_avg"
        print(f"Using transformed focal feature name: {focal_feature_col}")

    # Get model coefficients
    summary_df = model.summary()
    coefficients = summary_df["coef"]

    # Find the focal feature coefficient
    if focal_feature_col not in coefficients.index:
        raise ValueError(
            f"Focal feature '{focal_feature_col}' not found in model coefficients. Available features: {list(coefficients.index)}"
        )

    # Find interaction terms that include the focal feature
    interaction_cols = [
        col
        for col in coefficients.index
        if interaction_pattern in col
        and col.split(interaction_pattern)[0] == focal_feature_col
    ]

    # Calculate linear predictor from focal feature
    linear_predictor = coefficients[focal_feature_col] * X[focal_feature_col]

    # Add contribution from interaction terms
    for col in interaction_cols:
        parts = col.split(interaction_pattern)
        if len(parts) != 2:
            print(f"Warning: Unexpected interaction term format: {col}, skipping")
            continue

        other_feature = parts[1]
        if other_feature not in X.columns:
            print(
                f"Warning: Feature {other_feature} not found in data, skipping interaction {col}"
            )
            continue

        # For interaction terms, the contribution is β_interaction * (focal_feature * other_feature)
        # But X already has the interaction term as a column, so we can use it directly
        interaction_col = f"{focal_feature_col}{interaction_pattern}{other_feature}"
        if interaction_col in X.columns:
            linear_predictor += coefficients[col] * X[interaction_col]
        else:
            # Calculate the interaction term on the fly if it's not in X
            linear_predictor += coefficients[col] * (
                X[focal_feature_col] * X[other_feature]
            )

    # Convert to partial hazard ratio (exponential scale)
    return np.exp(linear_predictor)


def visualize_contribution_distribution(
    model: CoxPHModel,
    bins: int = 30,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    interaction_pattern: str = ":",
    return_figure: bool = False,
    verbose: bool = True,
) -> Union[pd.Series, Tuple[plt.Figure, pd.Series]]:
    """
    Analyze and visualize the distribution of feature contributions predicted by a Cox model.

    This function creates two visualizations:
    1. Feature contribution density plot
    2. Feature contributions by event status (histogram)

    Args:
        model (CoxPHModel): Fitted Cox model with interaction terms
        bins (int, optional): Number of bins for histograms. Defaults to 30.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 5).
        save_path (Optional[str], optional): If provided, saves the plot to this path.
        interaction_pattern (str, optional): Pattern used to identify interaction terms. Defaults to ":".
        return_figure (bool, optional): If True, returns the figure object instead of displaying it. Defaults to False.
        verbose (bool, optional): If True, prints detailed statistics to console. Defaults to True.

    Returns:
        Union[pd.Series, Tuple[plt.Figure, pd.Series]]: The calculated feature contributions, or a tuple of (figure, contributions) if return_figure is True
    """
    # Use data from the model
    X = model.X
    E = model.E

    # Use model's focal_feature if available
    focal_feature = model.focal_feature

    # Get a display name without the _avg suffix for plotting
    display_name = focal_feature
    if focal_feature and focal_feature.endswith("_avg"):
        display_name = focal_feature[:-4]  # Remove _avg suffix for display

    # Calculate feature contributions
    contributions = calculate_feature_contribution(
        model, X, focal_feature, interaction_pattern
    )

    # If focal_feature is still None, try to infer it from the model
    if focal_feature is None:
        interaction_cols = [
            col for col in model.summary().index if interaction_pattern in col
        ]
        if interaction_cols:
            focal_feature = interaction_cols[0].split(interaction_pattern)[0]
            display_name = focal_feature
            if display_name.endswith("_avg"):
                display_name = display_name[:-4]
        else:
            focal_feature = "feature"  # Default name if can't determine
            display_name = focal_feature

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame({"contribution": contributions})
    analysis_df["event"] = E.values

    # Create figure for visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 1. KDE plot of contributions (Contribution Density)
    contributions.plot.kde(ax=ax1, color="darkblue", linewidth=2)
    ax1.set_title(f"{display_name} Contribution Density")
    ax1.set_xlabel("Contribution to Linear Predictor")
    ax1.set_ylabel("Density")

    # Add vertical lines for different quantiles
    quantiles = [0.25, 0.5, 0.75]
    for q, color in zip(quantiles, ["orange", "green", "red"]):
        value = np.quantile(contributions, q)
        ax1.axvline(
            value,
            color=color,
            linestyle="--",
            linewidth=1,
            label=f"{q*100}%: {value:.4f}",
        )
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Histogram of contributions by event status
    event_mask = analysis_df["event"] == 1

    # Plot for events
    ax2.hist(
        analysis_df.loc[event_mask, "contribution"],
        bins=bins // 2,
        alpha=0.6,
        color="red",
        edgecolor="black",
        label="Events",
    )

    # Plot for non-events
    ax2.hist(
        analysis_df.loc[~event_mask, "contribution"],
        bins=bins // 2,
        alpha=0.6,
        color="orange",
        edgecolor="black",
        label="No Events",
    )

    ax2.set_title(f"{display_name} Contributions by Event Status")
    ax2.set_xlabel("Contribution to Linear Predictor")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Print basic statistics
    if verbose:
        print(f"\n--- {display_name} Contribution Analysis ---")
        print(f"Number of samples: {len(analysis_df)}")
        print(f"Mean: {analysis_df['contribution'].mean():.4f}")
        print(f"Median: {analysis_df['contribution'].median():.4f}")
        print(f"Standard deviation: {analysis_df['contribution'].std():.4f}")
        print(f"Minimum: {analysis_df['contribution'].min():.4f}")
        print(f"Maximum: {analysis_df['contribution'].max():.4f}")

        # Print quantiles
        quantile_values = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        quantiles = analysis_df["contribution"].quantile(quantile_values)
        # Convert to dictionary for type-safe access
        quantiles_dict = dict(zip(quantile_values, quantiles))
        print("\nQuantiles:")
        for q in quantile_values:
            val = quantiles_dict[q]
            print(f"  {q*100:2.0f}%: {val:.4f}")

        # Compare distributions for events vs non-events
        events = analysis_df[analysis_df["event"] == 1]["contribution"]
        non_events = analysis_df[analysis_df["event"] == 0]["contribution"]

        print("\nEvent-based statistics:")
        print(
            f"Samples with events: {len(events)} ({len(events)/len(analysis_df)*100:.2f}%)"
        )
        # Fix the negative sign bug in the output
        print(
            f"Samples without events: {len(non_events)} ({len(non_events)/len(analysis_df)*100:.2f}%)"
        )
        print(f"Mean contribution for events: {events.mean():.4f}")
        print(f"Mean contribution for non-events: {non_events.mean():.4f}")

        # Mann-Whitney U test (non-parametric test for differences in distribution)
        try:
            mw_u, mw_p = mannwhitneyu(events, non_events)
            print(f"Mann-Whitney U test: statistic={mw_u:.4f}, p-value={mw_p:.4e}")
        except Exception as e:
            print(f"Mann-Whitney U test failed: {e}")

        # T-test (parametric test for differences in means)
        try:
            t_stat, t_p = ttest_ind(events, non_events, equal_var=False)
            print(f"t-test: statistic={t_stat:.4f}, p-value={t_p:.4e}")
        except Exception as e:
            print(f"t-test failed: {e}")

    plt.tight_layout()

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    # Get the current figure before showing it
    fig = plt.gcf()

    # Only show the figure if not returning it
    if not return_figure:
        plt.show()

    if return_figure:
        return fig, contributions
    else:
        return contributions


def predict_contribution(
    self: CoxPHModel,
    X: Optional[pd.DataFrame] = None,
    focal_feature: Optional[str] = None,
    interaction_pattern: str = ":",
) -> pd.Series:
    """
    Calculate the contribution of a focal feature and its interactions to the model prediction.

    This is a convenience method added to the CoxPHModel class to allow easy calculation
    of feature contributions.

    Args:
        X (pd.DataFrame, optional): Feature data. If None, uses training data.
        focal_feature (str, optional): Name of the focal feature to calculate contribution for.
                                       If None, uses the focal_feature attribute from the model.
                                       Can use the original column name without the "_avg" suffix.
        interaction_pattern (str, optional): Pattern used to identify interaction terms.

    Returns:
        pd.Series: Net contribution of the focal feature and its interactions for each sample
                  in partial hazard form (exponential scale)
    """
    # If focal_feature not provided, use the model's focal_feature attribute
    if focal_feature is None:
        focal_feature = self.focal_feature

    return calculate_feature_contribution(
        model=self,
        X=X,
        focal_feature=focal_feature,
        interaction_pattern=interaction_pattern,
    )


# Monkey patch the CoxPHModel class to add the predict_contribution method
# Use setattr for better compatibility with type checking
setattr(CoxPHModel, "predict_contribution", predict_contribution)
