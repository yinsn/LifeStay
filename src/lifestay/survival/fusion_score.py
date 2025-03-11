"""
Fusion Score Module for Survival Analysis

This module provides functions for calculating an optimized fusion score
by combining risk scores and feature contributions to maximize the AUC
for event prediction in survival analysis.
"""

from typing import Dict, Optional, Tuple, Union

import pandas as pd
from lifelines.fitters.coxph_fitter import CoxPHFitter as CoxPHModel
from sklearn.metrics import roc_auc_score

from lifestay.survival.metrics_calculation import calculate_risk_metrics


def optimize_fusion_weights(
    df: pd.DataFrame,
    weight_steps: int = 100,
) -> Tuple[float, float, float]:
    """
    Optimize the weights for fusion score to maximize AUC.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'risk_score', 'contribution', and 'event' columns.
    weight_steps : int, optional
        Number of steps to try between 0 and 1 for weights, by default 100

    Returns
    -------
    Tuple[float, float, float]
        Optimal risk_weight, optimal contribution_weight, and the corresponding AUC value
    """
    best_auc = 0.0
    best_risk_weight = 0.5  # Default value
    best_contribution_weight = 0.5  # Default value

    # Try different weight combinations where risk_weight + contribution_weight = 1
    for i in range(weight_steps + 1):
        risk_weight = i / weight_steps
        contribution_weight = 1.0 - risk_weight

        # Calculate fusion score
        fusion_score = (
            risk_weight * df["risk_score"] + contribution_weight * df["contribution"]
        )

        # Calculate AUC for event prediction
        try:
            auc = roc_auc_score(df["event"], fusion_score)

            # Update best weights if AUC is improved
            if auc > best_auc:
                best_auc = auc
                best_risk_weight = risk_weight
                best_contribution_weight = contribution_weight
        except ValueError:
            # Handle case where all predictions or true values are the same
            continue

    return best_risk_weight, best_contribution_weight, best_auc


def get_optimal_fusion_formula(
    model: CoxPHModel,
    focal_feature: Optional[str] = None,
    interaction_pattern: str = ":",
    weight_steps: int = 100,
) -> Dict[str, Union[str, float, Dict[str, float]]]:
    """
    Calculate the optimal fusion formula for combining risk_score and contribution.

    Parameters
    ----------
    model : CoxPHModel
        The fitted Cox proportional hazards model.
    focal_feature : Optional[str], optional
        The focal feature for contribution analysis, by default None
    interaction_pattern : str, optional
        The pattern used to identify interaction terms, by default ":"
    weight_steps : int, optional
        Number of steps to try between 0 and 1 for weights, by default 100

    Returns
    -------
    Dict[str, Union[str, float, Dict[str, float]]]
        Dictionary containing:
        - 'formula': The formula string representation
        - 'auc': The AUC value achieved with optimal weights
        - 'weights': Dictionary with keys 'w1' and 'w2' for the optimal weights
          (w1 is risk_weight, w2 is contribution_weight)
    """
    # Calculate risk metrics
    risk_df = calculate_risk_metrics(
        model=model,
        focal_feature=focal_feature,
        interaction_pattern=interaction_pattern,
    )

    # Optimize weights
    risk_weight, contribution_weight, auc = optimize_fusion_weights(
        risk_df, weight_steps=weight_steps
    )

    # Create formula string
    formula = f"score = {risk_weight:.4f} * risk_score + {contribution_weight:.4f} * contribution"

    # For backward compatibility, still use 'w1' and 'w2' as keys in the weights dictionary
    return {
        "formula": formula,
        "auc": auc,
        "weights": {"w1": risk_weight, "w2": contribution_weight},
    }


def calculate_fusion_score(
    model: CoxPHModel,
    focal_feature: Optional[str] = None,
    interaction_pattern: str = ":",
    custom_weights: Optional[Dict[str, float]] = None,
    weight_steps: int = 100,
    print_metrics: bool = True,
    debug: bool = False,
    create_table: bool = False,
) -> Dict[str, Union[pd.DataFrame, Dict, float]]:
    """
    Calculate the fusion score using either optimized or custom weights and
    return comprehensive information about the model, weights, and features.

    Parameters
    ----------
    model : CoxPHModel
        The fitted Cox proportional hazards model.
    focal_feature : Optional[str], optional
        The focal feature for contribution analysis, by default None
    interaction_pattern : str, optional
        The pattern used to identify interaction terms, by default ":"
    custom_weights : Optional[Dict[str, float]], optional
        Custom weights to use instead of optimized ones.
        Should contain 'w1' and 'w2' keys. By default None
        (w1 is risk_weight, w2 is contribution_weight)
    weight_steps : int, optional
        Number of steps to try between 0 and 1 for weights, by default 100
    print_metrics : bool, optional
        Whether to print the AUC and formula to console, by default True
    debug : bool, optional
        Whether to print additional debug information, by default False
    create_table : bool, optional
        Whether to generate an HTML table with the fusion score results, by default False

    Returns
    -------
    Dict[str, Union[pd.DataFrame, Dict, float]]
        Dictionary containing:
        - 'data': Sorted DataFrame with fusion_score column
        - 'formula': Formula string representation
        - 'auc': AUC value for the fusion score
        - 'weights': Dictionary with fusion weights (w1 for risk_weight, w2 for contribution_weight)
        - 'risk_features': Dictionary with feature names and weights for risk component
        - 'contribution_features': Dictionary with feature names and weights for contribution component
    """
    # Try to get focal_feature from model if not provided
    if focal_feature is None and hasattr(model, "focal_feature"):
        focal_feature = model.focal_feature

    # Debug model info
    if debug:
        print("\nDEBUG Model Information:")
        print(f"Model type: {type(model).__name__}")
        print(
            f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}"
        )
        if hasattr(model, "summary"):
            try:
                print("Model has summary attribute/method")
                if callable(model.summary):
                    print("summary is callable")
                else:
                    print("summary is not callable")
            except Exception as e:
                print(f"Error checking summary: {e}")

        if hasattr(model, "basic_cox"):
            print("\nModel has basic_cox attribute")
            basic_cox = model.basic_cox
            print(f"basic_cox type: {type(basic_cox).__name__}")
            print(
                f"basic_cox attrs: {[attr for attr in dir(basic_cox) if not attr.startswith('_') and attr != 'summary']}"
            )

        if hasattr(model, "interaction_cox"):
            print("\nModel has interaction_cox attribute")
            int_cox = model.interaction_cox
            print(f"interaction_cox type: {type(int_cox).__name__}")
            print(
                f"interaction_cox attrs: {[attr for attr in dir(int_cox) if not attr.startswith('_') and attr != 'summary']}"
            )

    # Calculate risk metrics
    risk_df = calculate_risk_metrics(
        model=model,
        focal_feature=focal_feature,
        interaction_pattern=interaction_pattern,
    )

    # Get model coefficients for feature importance
    risk_features = {}
    contribution_features = {}

    try:
        # Direct access to coefficients if available
        if hasattr(model, "_coefficients") and model._coefficients is not None:
            if debug:
                print("\nFound _coefficients attribute")

            for feature, coef in model._coefficients.items():
                if interaction_pattern in feature or (
                    focal_feature is not None and focal_feature in feature
                ):
                    contribution_features[feature] = coef
                else:
                    risk_features[feature] = coef

        # Check model.model if available (sometimes used in wrapped models)
        elif hasattr(model, "model") and model.model is not None:
            if debug:
                print("\nFound model.model attribute")

            inner_model = model.model
            if hasattr(inner_model, "params_"):
                for feature, coef in inner_model.params_.items():
                    if interaction_pattern in feature or (
                        focal_feature is not None and focal_feature in feature
                    ):
                        contribution_features[feature] = coef
                    else:
                        risk_features[feature] = coef

        # Check if the model has a basic_cox attribute (for custom interaction models)
        elif hasattr(model, "basic_cox") and model.basic_cox is not None:
            if debug:
                print("\nExtracting from basic_cox and interaction_cox")

            # Get features from basic_cox
            basic_model = model.basic_cox
            if hasattr(basic_model, "params_"):
                for feature, coef in basic_model.params_.items():
                    risk_features[feature] = coef
            elif hasattr(basic_model, "hazards_"):
                features = basic_model.hazards_.columns.tolist()
                for feature in features:
                    risk_features[feature] = 1.0  # Using placeholder value

            # Get features from interaction_cox if available
            if hasattr(model, "interaction_cox") and model.interaction_cox is not None:
                interaction_model = model.interaction_cox
                if hasattr(interaction_model, "params_"):
                    for feature, coef in interaction_model.params_.items():
                        contribution_features[feature] = coef
                elif hasattr(interaction_model, "hazards_"):
                    features = interaction_model.hazards_.columns.tolist()
                    for feature in features:
                        contribution_features[feature] = 1.0  # Using placeholder value

        # If nothing found yet, try standard approaches
        if not risk_features and not contribution_features:
            if debug:
                print("\nTrying standard approaches")

            # Try to get coefficients directly from the model
            if hasattr(model, "params_"):
                if debug:
                    print("Model has params_ attribute")
                    print(f"params_ type: {type(model.params_)}")
                    print(
                        f"params_ content (first few): {list(model.params_.items())[:5] if hasattr(model.params_, 'items') else model.params_}"
                    )

                if hasattr(model.params_, "items"):
                    for feature, coef in model.params_.items():
                        # Separate features into risk and contribution
                        if interaction_pattern in str(feature) or (
                            focal_feature is not None and focal_feature in str(feature)
                        ):
                            contribution_features[str(feature)] = coef
                        else:
                            risk_features[str(feature)] = coef

            # If model has hazards_, try to extract feature names
            elif hasattr(model, "hazards_"):
                if debug:
                    print("Model has hazards_ attribute")

                features = model.hazards_.columns.tolist()
                for feature in features:
                    if interaction_pattern in feature or (
                        focal_feature is not None and focal_feature in feature
                    ):
                        contribution_features[feature] = 1.0  # Using placeholder
                    else:
                        risk_features[feature] = 1.0  # Using placeholder

        # If still empty, try one last approach
        if not risk_features and not contribution_features:
            if debug:
                print("\nTrying to access coefficients through model components")

            # Special handling for lifelines CoxPHFitter
            if hasattr(model, "_norm_mean"):
                if debug:
                    print("Detected lifelines CoxPHFitter structure")

                # Get feature names from _norm_mean
                if hasattr(model, "_norm_mean") and model._norm_mean is not None:
                    features = list(model._norm_mean.keys())
                    coefs = model.hazards_ if hasattr(model, "hazards_") else None

                    for feature in features:
                        coef = coefs.loc[0, feature] if coefs is not None else 1.0
                        if interaction_pattern in feature or (
                            focal_feature is not None and focal_feature in feature
                        ):
                            contribution_features[feature] = coef
                        else:
                            risk_features[feature] = coef

    except Exception as e:
        if debug:
            import traceback

            print(f"Error extracting coefficients: {e}")
            traceback.print_exc()
        else:
            print(f"Warning: Could not extract model coefficients: {e}")
            print("Feature importance information will not be available.")

    # Use provided weights or optimize
    if custom_weights is not None:
        risk_weight = custom_weights.get("w1", 0.5)
        contribution_weight = custom_weights.get("w2", 0.5)
        # Calculate AUC with custom weights
        fusion_score = (
            risk_weight * risk_df["risk_score"]
            + contribution_weight * risk_df["contribution"]
        )
        auc = roc_auc_score(risk_df["event"], fusion_score)
    else:
        risk_weight, contribution_weight, auc = optimize_fusion_weights(
            risk_df, weight_steps=weight_steps
        )

    # Create formula string
    formula = f"score = {risk_weight:.4f} * risk_score + {contribution_weight:.4f} * contribution"

    # Calculate fusion score
    risk_df["fusion_score"] = (
        risk_weight * risk_df["risk_score"]
        + contribution_weight * risk_df["contribution"]
    )

    # Sort by fusion score in descending order
    sorted_df = risk_df.sort_values("fusion_score", ascending=False)

    # Extract feature names from risk metrics if we couldn't get them from the model
    if (
        not risk_features
        and not contribution_features
        and "feature_contributions" in risk_df.columns
    ):
        if debug:
            print("\nAttempting to extract features from risk_df feature_contributions")

        # Try to extract feature names from feature_contributions
        try:
            # Sample a few rows and extract feature names from the contributions
            for _, row in risk_df.iloc[:5].iterrows():
                if isinstance(row.get("feature_contributions"), dict):
                    contrib_dict = row["feature_contributions"]
                    for feature, value in contrib_dict.items():
                        if interaction_pattern in feature or (
                            focal_feature is not None and focal_feature in feature
                        ):
                            if feature not in contribution_features:
                                contribution_features[feature] = value
                        else:
                            if feature not in risk_features:
                                risk_features[feature] = value
        except Exception as e:
            if debug:
                print(f"Error extracting from feature_contributions: {e}")

    # Print metrics if requested
    if print_metrics:
        print(f"Fusion formula: {formula}")
        print(f"AUC: {auc:.4f}")
        print(f"Risk features: {len(risk_features)}")
        print(f"Contribution features: {len(contribution_features)}")

        if len(risk_features) > 0:
            print("\nRisk features (top 5):")
            for i, (feature, coef) in enumerate(
                sorted(risk_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            ):
                print(f"  {feature}: {coef}")

        if len(contribution_features) > 0:
            print("\nContribution features (top 5):")
            for i, (feature, coef) in enumerate(
                sorted(
                    contribution_features.items(), key=lambda x: abs(x[1]), reverse=True
                )[:5]
            ):
                print(f"  {feature}: {coef}")

    # Return comprehensive results
    result = {
        "data": sorted_df,
        "formula": formula,
        "auc": auc,
        "weights": {"w1": risk_weight, "w2": contribution_weight},
        "risk_features": risk_features,
        "contribution_features": contribution_features,
    }

    # Generate HTML table if requested
    if create_table:
        generate_html_table(result)

    return result


def generate_html_table(result: Dict[str, Union[pd.DataFrame, Dict, float]]) -> None:
    """
    Generate an HTML file with a table showing the fusion score components.

    Parameters
    ----------
    result : Dict[str, Union[pd.DataFrame, Dict, float]]
        The result dictionary from calculate_fusion_score
    """
    import os
    from datetime import datetime
    from typing import Dict, cast

    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fusion Score Analysis</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .metrics {{
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            table, th, td {{
                border: 1px solid #ddd;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .footer {{
                margin-top: 30px;
                font-size: 0.8em;
                color: #777;
            }}
        </style>
    </head>
    <body>
        <h1>Fusion Score Analysis</h1>

        <div class="metrics">
            <h2>Key Metrics</h2>
            <p><strong>Formula:</strong> {result['formula']}</p>
            <p><strong>AUC:</strong> {result['auc']:.6f}</p>
        </div>

        <h2>Component Details</h2>
        <table>
            <tr>
                <th>Component</th>
                <th>Key</th>
                <th>Value</th>
            </tr>
    """

    # Add weights section if it exists and is a dictionary
    if "weights" in result and isinstance(result["weights"], dict):
        weights = cast(Dict[str, float], result["weights"])
        html_content += """
            <tr>
                <td rowspan="2"><strong>weights</strong></td>
        """

        # Add risk_weight and contribution_weight (formerly w1 and w2)
        if "w1" in weights:
            risk_weight_row = f"""
                <td>risk_weight</td>
                <td>{weights['w1']:.4f}</td>
            </tr>
            """
            html_content += risk_weight_row

        if "w2" in weights:
            contribution_weight_row = f"""
            <tr>
                <td>contribution_weight</td>
                <td>{weights['w2']:.4f}</td>
            </tr>
            """
            html_content += contribution_weight_row

    # Add risk_features section if it exists and is a dictionary
    if "risk_features" in result and isinstance(result["risk_features"], dict):
        risk_features = cast(Dict[str, float], result["risk_features"])
        if risk_features:  # Only proceed if the dictionary is not empty
            html_content += f"""
                <tr>
                    <td rowspan="{len(risk_features)}"><strong>risk_features</strong></td>
            """

            # Add risk feature rows
            first_risk = True
            for feature, value in risk_features.items():
                if first_risk:
                    html_content += f"""
                        <td>{feature}</td>
                        <td>{value:.10e}</td>
                    </tr>
                    """
                    first_risk = False
                else:
                    html_content += f"""
                    <tr>
                        <td>{feature}</td>
                        <td>{value:.10e}</td>
                    </tr>
                    """

    # Add contribution_features section if it exists and is a dictionary
    if "contribution_features" in result and isinstance(
        result["contribution_features"], dict
    ):
        contribution_features = cast(Dict[str, float], result["contribution_features"])
        if contribution_features:  # Only proceed if the dictionary is not empty
            html_content += f"""
                <tr>
                    <td rowspan="{len(contribution_features)}"><strong>contribution_features</strong></td>
            """

            # Add contribution feature rows
            first_contrib = True
            for feature, value in contribution_features.items():
                if first_contrib:
                    html_content += f"""
                        <td>{feature}</td>
                        <td>{value:.10e}</td>
                    </tr>
                    """
                    first_contrib = False
                else:
                    html_content += f"""
                    <tr>
                        <td>{feature}</td>
                        <td>{value:.10e}</td>
                    </tr>
                    """

    html_content += f"""
        </table>

    </body>
    </html>
    """

    # Determine the output file path
    output_dir = "fusion_score_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"fusion_score_{timestamp}.html")

    # Write HTML to file
    with open(output_file, "w") as f:
        f.write(html_content)

    print(f"HTML table generated: {output_file}")
