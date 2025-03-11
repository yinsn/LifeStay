"""
Survival Analysis Module for lifestay

This module provides tools for survival analysis, including:
- Data conversion utilities to prepare data for survival analysis
- Implementation of survival models (Cox Proportional Hazards, etc.)
- Evaluation metrics and visualization functions
- Risk stratification and patient grouping
- Advanced modeling techniques with feature selection and model comparison
- Feature contribution analysis for understanding feature impact on risk
- Comprehensive visualization reports combining multiple analyses

The module is designed to work with data prepared by SampleBuilder.
"""

# Re-enabling the import now that the file exists
from .basic_cox import basic_cox_model
from .comprehensive_visualization import generate_comprehensive_report
from .contribution_analysis import (
    calculate_feature_contribution,
    visualize_contribution_distribution,
)
from .cox_model import CoxPHModel
from .data_builder import SurvivalDatasetBuilder
from .data_converter import convert_to_lifelines_format
from .interaction_cox import cox_model_with_interactions
from .metrics_calculation import calculate_risk_metrics
from .risk_stratification import survival_risk_groups, visualize_risk_distribution
from .utils import (
    create_dataset,
    create_survival_dataset_from_sample_builder,
    example_usage,
    fit_and_evaluate_cox_model,
    plot_survival_curves,
)

__all__ = [
    "advanced_cox_modeling",
    "basic_cox_model",
    "calculate_feature_contribution",
    "calculate_risk_metrics",
    "convert_to_lifelines_format",
    "cox_model_with_interactions",
    "CoxPHModel",
    "create_dataset",
    "create_survival_dataset_from_sample_builder",
    "example_usage",
    "fit_and_evaluate_cox_model",
    "generate_comprehensive_report",
    "plot_survival_curves",
    "survival_risk_groups",
    "SurvivalDatasetBuilder",
    "visualize_contribution_distribution",
    "visualize_risk_distribution",
]
