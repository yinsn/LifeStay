"""
Risk Stratification Module for Survival Analysis

This module provides functions for risk stratification and risk grouping based on
survival analysis models, primarily Cox Proportional Hazards models.
"""

from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

from .basic_cox import basic_cox_model
from .cox_model import CoxPHModel
from .interaction_cox import cox_model_with_interactions

# Re-export these functions for backward compatibility
__all__ = [
    "basic_cox_model",
    "cox_model_with_interactions",
    "survival_risk_groups",
    "visualize_risk_distribution",
]


def survival_risk_groups(
    model: CoxPHModel, n_groups: int = 3
) -> Dict[str, pd.DataFrame]:
    """
    Stratify patients into risk groups based on the Cox model and visualize survival curves.

    Args:
        model (CoxPHModel): Fitted Cox model
        n_groups (int, optional): Number of risk groups to create. Defaults to 3.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrames for each risk group
    """
    print(f"\n--- Risk Stratification ({n_groups} groups) ---")

    # Get data from the model
    X = model.X
    T = model.T
    E = model.E

    # Predict risk scores
    risk_scores = model.predict_risk(X)

    # Print basic statistics of risk scores for debugging
    print(
        f"Risk scores - min: {risk_scores.min():.4f}, max: {risk_scores.max():.4f}, mean: {risk_scores.mean():.4f}"
    )
    print(
        f"Risk scores distribution - 25%: {np.percentile(risk_scores, 25):.4f}, 50%: {np.percentile(risk_scores, 50):.4f}, 75%: {np.percentile(risk_scores, 75):.4f}"
    )

    # Check if all risk scores are identical
    if risk_scores.min() == risk_scores.max():
        print(
            "All risk scores are identical, cannot group based on risk scores. Using direct grouping method."
        )

        # Separate samples with events and without events
        event_mask = E == 1
        event_samples = X[event_mask].index
        non_event_samples = X[~event_mask].index

        total_events = len(event_samples)
        total_non_events = len(non_event_samples)

        print(
            f"Total samples: {len(X)}, Samples with events: {total_events}, Samples without events: {total_non_events}"
        )

        if total_events == 0:
            print("No events in the dataset, using survival time for grouping.")
            # For datasets without events, group by survival time
            try:
                risk_groups = pd.qcut(-T, q=n_groups, labels=False)
            except ValueError:
                print("Survival time grouping failed, using uniform distribution.")
                # Uniform distribution
                risk_groups = pd.Series(
                    np.repeat(range(n_groups), len(X) // n_groups + 1)[: len(X)],
                    index=X.index,
                )
        else:
            # Number of event samples per group
            events_per_group = [total_events // n_groups] * n_groups
            # Handle remainder
            for i in range(total_events % n_groups):
                events_per_group[i] += 1

            # Sort event samples by survival time (shorter survival time indicates higher risk)
            sorted_event_samples = T.loc[event_samples].sort_values().index

            # Sort non-event samples by survival time
            sorted_non_event_samples = T.loc[non_event_samples].sort_values().index

            # Number of non-event samples per group
            non_events_per_group = [total_non_events // n_groups] * n_groups
            # Handle remainder
            for i in range(total_non_events % n_groups):
                non_events_per_group[i] += 1

            # Initialize risk groups Series
            risk_groups = pd.Series(index=X.index, dtype=int)

            # Assign event and non-event samples to each group
            start_event_idx = 0
            start_non_event_idx = 0

            for group in range(n_groups):
                # Assign event samples
                end_event_idx = start_event_idx + events_per_group[group]
                group_event_samples = sorted_event_samples[
                    start_event_idx:end_event_idx
                ]
                risk_groups.loc[group_event_samples] = group
                start_event_idx = end_event_idx

                # Assign non-event samples
                end_non_event_idx = start_non_event_idx + non_events_per_group[group]
                group_non_event_samples = sorted_non_event_samples[
                    start_non_event_idx:end_non_event_idx
                ]
                risk_groups.loc[group_non_event_samples] = group
                start_non_event_idx = end_non_event_idx

            print(
                f"Successfully distributed event samples evenly across {n_groups} risk groups"
            )
    else:
        # Use standard risk score grouping method
        try:
            risk_groups = pd.qcut(risk_scores, q=n_groups, labels=False)
            print("Using equal-frequency binning (qcut)")
        except ValueError as e:
            print(f"Equal-frequency binning failed with error: {e}")
            print("Falling back to equal-width binning (cut)")

            try:
                risk_groups = pd.cut(risk_scores, bins=n_groups, labels=False)
            except ValueError:
                print("Equal-width binning also failed. Dividing samples evenly.")
                n_samples = len(X)
                group_size = n_samples // n_groups
                risk_groups = pd.Series(
                    np.repeat(range(n_groups), group_size)[:n_samples], index=X.index
                )

    # Convert risk groups to integer Series (if not already)
    risk_groups = pd.Series(risk_groups, index=X.index)

    # Describe characteristics of each group
    print("\nRisk group characteristics:")
    for group in range(n_groups):
        mask = risk_groups == group
        group_size = mask.sum()
        group_events = E[mask].sum()
        group_mean_time = T[mask].mean()
        event_rate = group_events / group_size if group_size > 0 else 0

        print(
            f"Risk Group {group}: {group_size} samples, {group_events} events, "
            f"{event_rate:.2%} event rate, mean survival time: {group_mean_time:.2f}"
        )

    # Plot Kaplan-Meier curves for each risk group
    print("\nPlotting Kaplan-Meier curves for risk groups...")
    try:
        # Import KaplanMeierFitter
        from lifelines import KaplanMeierFitter

        plt.figure(figsize=(10, 6))

        # Use different colors and styles
        # Create a list of colors for the groups
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ][:n_groups]

        # Create separate KaplanMeierFitter instances for each risk group
        for group in range(n_groups):
            mask = risk_groups == group
            if mask.sum() > 0:
                kmf = KaplanMeierFitter()
                kmf.fit(T[mask], E[mask], label=f"Risk Group {group}")
                kmf.plot_survival_function(
                    ax=plt.gca(), ci_show=True, color=colors[group]
                )

        plt.title("Kaplan-Meier Curves by Risk Group")
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Add log curve plot (log-rank test)
        from lifelines.statistics import logrank_test

        if n_groups > 1:
            print("\nLog-rank test between risk groups:")
            for i in range(n_groups):
                for j in range(i + 1, n_groups):
                    mask_i = risk_groups == i
                    mask_j = risk_groups == j

                    if mask_i.sum() > 0 and mask_j.sum() > 0:
                        results = logrank_test(
                            T[mask_i], T[mask_j], E[mask_i], E[mask_j]
                        )
                        p_value = results.p_value
                        print(
                            f"Risk Group {i} vs Risk Group {j}: p-value = {p_value:.4f}"
                        )
    except Exception as e:
        print(f"Couldn't plot Kaplan-Meier curves due to error: {e}")

    # Return patient data after grouping
    return {f"risk_group_{group}": X[risk_groups == group] for group in range(n_groups)}


def visualize_risk_distribution(
    model: CoxPHModel,
    bins: int = 30,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    return_figure: bool = False,
    verbose: bool = True,
) -> Union[pd.Series, Tuple[plt.Figure, pd.Series]]:
    """
    Analyze and visualize the distribution of risk scores predicted by a Cox model.

    This function creates two visualizations:
    1. Risk score density plot
    2. Risk scores by event status (histogram)

    Args:
        model (CoxPHModel): Fitted Cox model
        bins (int, optional): Number of bins for histograms. Defaults to 30.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 5).
        save_path (Optional[str], optional): If provided, saves the plot to this path.
                                            Defaults to None.
        return_figure (bool, optional): If True, returns the figure object instead of displaying it. Defaults to False.
        verbose (bool, optional): If True, prints detailed statistics to console. Defaults to True.

    Returns:
        Union[pd.Series, Tuple[plt.Figure, pd.Series]]: The calculated risk scores, or a tuple of (figure, risk_scores) if return_figure is True
    """
    # Get data from the model
    X = model.X
    E = model.E

    # Predict risk scores
    risk_scores = model.predict_risk(X)

    # Create a DataFrame for analysis that includes risk scores
    analysis_df = pd.DataFrame({"risk_score": risk_scores})

    # Add event indicators
    analysis_df["event"] = E.values

    # Create figure for visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 1. KDE plot of risk scores (Risk Score Density)
    risk_scores.plot.kde(ax=ax1, color="darkblue", linewidth=2)
    ax1.set_title("Risk Score Density")
    ax1.set_xlabel("Risk Score")
    ax1.set_ylabel("Density")

    # Add vertical lines for different quantiles
    quantiles = [0.25, 0.5, 0.75]
    for q, color in zip(quantiles, ["orange", "green", "red"]):
        value = np.quantile(risk_scores, q)
        ax1.axvline(
            value,
            color=color,
            linestyle="--",
            linewidth=1,
            label=f"{q*100}%: {value:.4f}",
        )
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Histogram of risk scores by event status
    event_mask = analysis_df["event"] == 1

    # Plot for events
    ax2.hist(
        analysis_df.loc[event_mask, "risk_score"],
        bins=bins // 2,
        alpha=0.6,
        color="red",
        edgecolor="black",
        label="Events",
    )

    # Plot for non-events
    ax2.hist(
        analysis_df.loc[~event_mask, "risk_score"],
        bins=bins // 2,
        alpha=0.6,
        color="orange",
        edgecolor="black",
        label="No Events",
    )

    ax2.set_title("Risk Scores by Event Status")
    ax2.set_xlabel("Risk Score")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Print basic statistics
    if verbose:
        print("\n--- Risk Score Distribution Analysis ---")
        print(f"Number of samples: {len(analysis_df)}")
        print(f"Mean: {analysis_df['risk_score'].mean():.4f}")
        print(f"Median: {analysis_df['risk_score'].median():.4f}")
        print(f"Standard deviation: {analysis_df['risk_score'].std():.4f}")
        print(f"Minimum: {analysis_df['risk_score'].min():.4f}")
        print(f"Maximum: {analysis_df['risk_score'].max():.4f}")

        # Print quantiles
        quantile_values = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        quantiles = analysis_df["risk_score"].quantile(quantile_values)
        # Convert to dictionary for type-safe access
        quantiles_dict = dict(zip(quantile_values, quantiles))
        print("\nQuantiles:")
        for q in quantile_values:
            val = quantiles_dict[q]
            print(f"  {q*100:2.0f}%: {val:.4f}")

        # Compare distributions for events vs non-events
        events = analysis_df[analysis_df["event"] == 1]["risk_score"]
        non_events = analysis_df[analysis_df["event"] == 0]["risk_score"]

        print("\nEvent-based statistics:")
        print(
            f"Samples with events: {len(events)} ({len(events)/len(analysis_df)*100:.2f}%)"
        )
        # Fix the negative sign bug in the output
        print(
            f"Samples without events: {len(non_events)} ({len(non_events)/len(analysis_df)*100:.2f}%)"
        )
        print(f"Mean risk score for events: {events.mean():.4f}")
        print(f"Mean risk score for non-events: {non_events.mean():.4f}")

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
        return fig, risk_scores
    else:
        return risk_scores
