"""
Comprehensive Visualization Module for Survival Analysis

This module provides tools for generating comprehensive visualization reports
that combine multiple survival analysis visualizations and metrics in a single HTML file.
"""

import io
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd

from .contribution_analysis import visualize_contribution_distribution
from .cox_model import CoxPHModel
from .fusion_score import calculate_fusion_score
from .risk_comparison_visualization import RiskComparisonVisualizer
from .risk_stratification import visualize_risk_distribution


def generate_comprehensive_report(
    model: CoxPHModel,
    output_path: Optional[str] = None,
    include_risk_distribution: bool = True,
    include_contribution_distribution: bool = True,
    include_3d_visualization: bool = True,
    include_fusion_score: bool = True,
    risk_plot_figsize: Tuple[int, int] = (12, 5),
    contribution_plot_figsize: Tuple[int, int] = (12, 5),
    plot_3d_height: int = 700,
    plot_3d_width: int = 900,
    custom_title: Optional[str] = None,
    fusion_weight_steps: int = 100,
    risk_bins: int = 30,
    verbose: bool = False,
) -> Union[str, Dict[str, Union[pd.DataFrame, Dict, float]]]:
    """
    Generate a comprehensive HTML report that combines multiple visualizations and metrics
    from a Cox Proportional Hazards model with interactions.

    Parameters
    ----------
    model : CoxPHModel
        A fitted Cox Proportional Hazards model with interactions.
    output_path : Optional[str], default=None
        Path to save the HTML report. If None, returns the HTML as a string.
    include_risk_distribution : bool, default=True
        Whether to include risk distribution visualizations.
    include_contribution_distribution : bool, default=True
        Whether to include contribution distribution visualizations.
    include_3d_visualization : bool, default=True
        Whether to include 3D risk comparison visualization.
    include_fusion_score : bool, default=True
        Whether to include fusion score analysis.
    risk_plot_figsize : Tuple[int, int], default=(12, 5)
        Figure size for risk distribution plots.
    contribution_plot_figsize : Tuple[int, int], default=(12, 5)
        Figure size for contribution distribution plots.
    plot_3d_height : int, default=700
        Height of the 3D visualization plot.
    plot_3d_width : int, default=900
        Width of the 3D visualization plot.
    custom_title : Optional[str], default=None
        Custom title for the report. If None, a default title is used.
    fusion_weight_steps : int, default=100
        Number of steps for optimizing fusion weights.
    risk_bins : int, default=30
        Number of bins for risk distribution visualizations.
    verbose : bool, default=False
        Whether to print detailed information to the console during processing.

    Returns
    -------
    Union[str, Dict[str, Union[pd.DataFrame, Dict, float]]]
        If output_path is None, returns the HTML content of the report as a string.
        If output_path is provided, returns the fusion score result dictionary.

    Examples
    --------
    >>> from lifestay.survival import CoxPHModel, generate_comprehensive_report
    >>> # Assuming 'model' is a fitted CoxPHModel with interactions
    >>> report_path = generate_comprehensive_report(
    ...     model=model,
    ...     output_path="comprehensive_report.html"
    ... )
    """
    if not model.fitted:
        raise ValueError("The model must be fitted before generating a report.")

    if not model.focal_feature:
        raise ValueError(
            "The model must have a focal feature for comprehensive visualization."
        )

    html_parts = []

    # Add report title
    title = (
        custom_title or f"Comprehensive Survival Analysis Report: {model.focal_feature}"
    )
    html_parts.append(f"<h1 style='text-align:center;'>{title}</h1>")

    # Add model information section
    html_parts.append("<div style='margin: 20px 0;'>")
    html_parts.append("<h2>Model Information</h2>")

    # Create model summary table
    summary_df = model.summary()
    model_info = {
        "Focal Feature": model.focal_feature or "None",
        "Heartbeat Feature": getattr(model, "heartbeat_column", None) or "None",
        "Window Size": getattr(model, "window_size", None) or "Not specified",
        "Sample Size": getattr(model, "sample_size", None)
        or len(model.model.event_observed),
        "Number of Samples": len(model.model.event_observed),
        "Number of Events": sum(model.model.event_observed),
        "Number of Features": len(summary_df),
        "Concordance Index": f"{model.evaluate().get('concordance_index', 0):.4f}",
    }

    html_parts.append("<table style='width:100%; border-collapse: collapse;'>")
    for key, value in model_info.items():
        html_parts.append(
            f"<tr><td style='border:1px solid #ddd; padding:8px; font-weight:bold;'>{key}</td>"
        )
        html_parts.append(
            f"<td style='border:1px solid #ddd; padding:8px;'>{value}</td></tr>"
        )
    html_parts.append("</table>")
    html_parts.append("</div>")

    # Add risk distribution visualization if requested
    if include_risk_distribution:
        html_parts.append("<div style='margin: 20px 0;'>")
        html_parts.append("<h2>Risk Score Distribution</h2>")
        html_parts.append(
            "<p>Visualization of risk score distribution for the Cox Proportional Hazards model.</p>"
        )

        # Generate risk distribution visualization with verbose flag
        risk_fig, risk_scores = visualize_risk_distribution(
            model=model,
            bins=risk_bins,
            figsize=risk_plot_figsize,
            save_path=None,  # We'll capture the figure output directly
            return_figure=True,
            verbose=verbose,  # Pass verbose flag to suppress console output
        )

        # Convert matplotlib figure to HTML
        buffer = io.StringIO()
        risk_fig.savefig(buffer, format="svg", bbox_inches="tight")
        plt.close(risk_fig)  # Close the figure to free memory
        html_parts.append(buffer.getvalue())

        html_parts.append("</div>")

    # Add contribution distribution visualization if requested
    if include_contribution_distribution:
        html_parts.append("<div style='margin: 20px 0;'>")
        html_parts.append("<h2>Feature Contribution Distribution</h2>")
        html_parts.append(
            "<p>Visualization of feature contribution distribution for the interaction model.</p>"
        )

        # Generate contribution distribution visualization with verbose flag
        contrib_fig, contributions = visualize_contribution_distribution(
            model=model,
            figsize=contribution_plot_figsize,
            return_figure=True,
            verbose=verbose,  # Pass verbose flag to suppress console output
        )

        # Convert matplotlib figure to HTML
        buffer = io.StringIO()
        contrib_fig.savefig(buffer, format="svg", bbox_inches="tight")
        plt.close(contrib_fig)  # Close the figure to free memory
        html_parts.append(buffer.getvalue())

        html_parts.append("</div>")

    # Add 3D visualization if requested
    if include_3d_visualization:
        html_parts.append("<div style='margin: 20px 0;'>")
        html_parts.append("<h2>3D Risk Comparison Visualization</h2>")
        html_parts.append(
            "<p>Interactive 3D visualization of risk relationships between features.</p>"
        )

        # Generate 3D visualization
        visualizer = RiskComparisonVisualizer().with_model(model)
        fig = visualizer.visualize(
            plot_type="3d_surface",
            height=plot_3d_height,
            width=plot_3d_width,
        )

        # Add Plotly figure to HTML
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

        html_parts.append("</div>")

    # Add fusion score analysis if requested
    fusion_score_result = None
    if include_fusion_score:
        html_parts.append("<div style='margin: 20px 0;'>")
        html_parts.append("<h2>Fusion Score Analysis</h2>")
        html_parts.append(
            "<p>Analysis of optimal feature weights for risk prediction.</p>"
        )

        # Generate fusion score analysis with verbose flag
        fusion_score_result = calculate_fusion_score(
            model=model,
            weight_steps=fusion_weight_steps,
            create_table=False,  # We'll handle the table creation ourselves
            print_metrics=verbose,  # Pass verbose flag to suppress console output
            debug=verbose,  # Pass verbose flag to suppress console output
        )

        # Extract fusion score table content
        fusion_table_html = create_fusion_score_section(fusion_score_result)
        html_parts.append(fusion_table_html)

        html_parts.append("</div>")

    # Combine all parts into a single HTML document
    style = """
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #2980b9;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }
        h3 {
            color: #3498db;
            margin-top: 20px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            text-align: left;
            padding: 8px;
            border: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .metrics {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .math-container {
            width: 90%;
            margin: 10px auto;
            overflow-x: auto;
            padding: 10px;
            background-color: #f8f8f8;
            border-radius: 5px;
        }
        .footer {
            margin-top: 50px;
            text-align: center;
            font-size: 0.8em;
            color: #7f8c8d;
        }
    </style>
    """

    # Add MathJax for LaTeX rendering
    mathjax = """
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    """

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{custom_title or "Comprehensive Survival Analysis Report"}</title>
    {style}
    {mathjax}
</head>
<body>
    <div class="container">
        {''.join(html_parts)}
        <div class="footer">
            <p>Generated using YIN's Survival Analysis Module</p>
        </div>
    </div>
</body>
</html>
"""

    if output_path:
        # Save HTML content to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        # Return the fusion score result dictionary if it exists, otherwise return an empty dict
        if fusion_score_result is not None:
            return fusion_score_result
        else:
            return {}  # Return empty dict instead of None to match return type
    else:
        return html_content


def create_fusion_score_section(
    result: Dict[str, Union[pd.DataFrame, Dict, float]],
) -> str:
    """
    Create an HTML section displaying the fusion score results.

    Parameters
    ----------
    result : Dict[str, Union[pd.DataFrame, Dict, float]]
        The result dictionary from calculate_fusion_score

    Returns
    -------
    str
        HTML content for the fusion score section
    """
    html_parts = []

    # Add the formula and key metrics
    html_parts.append("<div class='metrics'>")
    html_parts.append("<h3>Key Metrics</h3>")

    # Helper function to format feature names for LaTeX
    def format_feature_for_latex(feature_name: str) -> str:
        """Convert a feature name to LaTeX-compatible format with proper subscripts."""
        # Handle interaction terms (containing :)
        if ":" in feature_name:
            # Split by colon and format each part separately
            terms = feature_name.split(":")
            formatted_terms = [format_feature_for_latex(term) for term in terms]
            return r" \cdot ".join(formatted_terms)  # Join with multiplication symbol

        # Handle regular terms with underscores
        if "_" in feature_name:
            # Split by underscores
            parts = feature_name.split("_")

            # Process the main part (first element)
            main_text = parts[0]
            if main_text.startswith("ext"):
                # If it starts with 'ext', remove it and use the rest as main text
                main_text = main_text[3:]  # e.g. 'extlive' -> 'live'

            # Create LaTeX code with proper subscripts
            # Put all parts except the last one in \text{}
            if len(parts) > 1:
                # Join all parts except the last one with underscores
                main_parts = "_".join(parts[:-1])
                result = r"\text{" + main_parts + "}"
                # Add the last part as a subscript
                result += "_{" + parts[-1] + "}"
            else:
                result = r"\text{" + main_text + "}"

            return result

        # If no special formatting needed, wrap in \text{}
        return r"\text{" + feature_name + "}"

    # Add formula with MathJax
    if "formula" in result:
        # Extract weights with type checking
        weights = result.get("weights", {})
        if not isinstance(weights, dict):
            weights = {}
        risk_weight = weights.get("w1", 0.5)
        contribution_weight = weights.get("w2", 0.5)

        # Create MathJax formula with 3 significant digits
        html_parts.append("<div class='math-container'>")
        html_parts.append("<p><strong>Formula:</strong></p>")
        html_parts.append(
            r"$$\text{fusion score} = "
            + f"{risk_weight:.3g}"
            + r" \cdot \text{risk score} + "
            + f"{contribution_weight:.3g}"
            + r" \cdot \text{contribution}$$"
        )
        html_parts.append("</div>")

    # Add calculation methods using MathJax
    html_parts.append("<h3>Calculation Methods</h3>")

    # Risk score calculation formula
    html_parts.append("<div class='math-container'>")
    html_parts.append("<p><strong>Risk Score Calculation:</strong></p>")

    # Get risk features and sort by absolute coefficient value
    risk_features = result.get("risk_features", {})
    if not isinstance(risk_features, dict):
        risk_features = {}
    sorted_risk_features = sorted(
        risk_features.items(), key=lambda x: abs(x[1]), reverse=True
    )

    # Create the risk score formula with actual coefficients (3 sig digits)
    if sorted_risk_features:
        risk_formula = r"$$\text{risk score} = \exp\left("
        terms = []

        # Process each term, handling positive and negative coefficients
        for i, (feature, coef) in enumerate(sorted_risk_features):
            # Format with 3 significant digits, handling scientific notation
            if abs(coef) < 0.001 or abs(coef) >= 1000:
                # Use scientific notation for very small or large numbers
                coef_str = f"{abs(coef):.3e}".replace("e-0", "e-").replace("e+0", "e+")
                # Convert to LaTeX scientific notation
                if "e-" in coef_str:
                    base, exp = coef_str.split("e-")
                    coef_str = f"{base} \\times 10^{{-{exp}}}"
                elif "e+" in coef_str:
                    base, exp = coef_str.split("e+")
                    coef_str = f"{base} \\times 10^{{{exp}}}"
            else:
                # Use regular notation for normal-sized numbers
                coef_str = f"{abs(coef):.3g}"

            # First term doesn't need a sign prefix
            if i == 0:
                prefix = "-" if coef < 0 else ""
            else:
                # Add appropriate sign for subsequent terms
                prefix = " - " if coef < 0 else " + "

            # Format the feature name with proper LaTeX subscripts
            latex_feature = format_feature_for_latex(feature)
            terms.append(f"{prefix}{coef_str} \\cdot {latex_feature}")

        risk_formula += "".join(terms)
        risk_formula += r"\right)$$"
        html_parts.append(risk_formula)
    else:
        html_parts.append(
            r"$$\text{risk score} = \exp\left(\sum_{i} \beta_i \cdot x_i\right)$$"
        )

    html_parts.append("</div>")

    # Contribution calculation formula
    html_parts.append("<div class='math-container'>")
    html_parts.append("<p><strong>Contribution Calculation:</strong></p>")

    # Get contribution features and sort by absolute coefficient value
    contribution_features = result.get("contribution_features", {})
    if not isinstance(contribution_features, dict):
        contribution_features = {}
    sorted_contribution_features = sorted(
        contribution_features.items(), key=lambda x: abs(x[1]), reverse=True
    )

    # Create the contribution formula with actual coefficients (3 sig digits)
    if sorted_contribution_features:
        contrib_formula = r"$$\text{contribution} = \exp\left("
        terms = []

        # Process each term, handling positive and negative coefficients
        for i, (feature, coef) in enumerate(sorted_contribution_features):
            # Format with 3 significant digits, handling scientific notation
            if abs(coef) < 0.001 or abs(coef) >= 1000:
                # Use scientific notation for very small or large numbers
                coef_str = f"{abs(coef):.3e}".replace("e-0", "e-").replace("e+0", "e+")
                # Convert to LaTeX scientific notation
                if "e-" in coef_str:
                    base, exp = coef_str.split("e-")
                    coef_str = f"{base} \\times 10^{{-{exp}}}"
                elif "e+" in coef_str:
                    base, exp = coef_str.split("e+")
                    coef_str = f"{base} \\times 10^{{{exp}}}"
            else:
                # Use regular notation for normal-sized numbers
                coef_str = f"{abs(coef):.3g}"

            # First term doesn't need a sign prefix
            if i == 0:
                prefix = "-" if coef < 0 else ""
            else:
                # Add appropriate sign for subsequent terms
                prefix = " - " if coef < 0 else " + "

            # Format the feature name with proper LaTeX subscripts
            latex_feature = format_feature_for_latex(feature)
            terms.append(f"{prefix}{coef_str} \\cdot {latex_feature}")

        contrib_formula += "".join(terms)
        contrib_formula += r"\right)$$"
        html_parts.append(contrib_formula)
    else:
        html_parts.append(
            r"$$\text{contribution} = \exp\left(\sum_{j} \gamma_j \cdot z_j\right)$$"
        )

    html_parts.append("</div>")

    if "auc" in result:
        html_parts.append(f"<p><strong>AUC:</strong> {result.get('auc', 0):.6f}</p>")

    if "metrics" in result and isinstance(result["metrics"], dict):
        metrics = result["metrics"]
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                html_parts.append(
                    f"<p><strong>{metric_name}:</strong> {metric_value:.6f}</p>"
                )
            else:
                html_parts.append(
                    f"<p><strong>{metric_name}:</strong> {metric_value}</p>"
                )

    html_parts.append("</div>")

    # Add the weights table
    html_parts.append("<h3>Feature Weights</h3>")
    html_parts.append("<table>")
    html_parts.append("<tr><th>Component</th><th>Weight</th></tr>")

    if "weights" in result and isinstance(result["weights"], dict):
        weights = result["weights"]
        for feature, weight in weights.items():
            display_name = (
                "Risk Weight"
                if feature == "w1"
                else "Contribution Weight" if feature == "w2" else feature
            )
            html_parts.append(f"<tr><td>{display_name}</td><td>{weight:.6f}</td></tr>")

    html_parts.append("</table>")

    # Add risk features table if available, sorted by coefficient descending
    if (
        "risk_features" in result
        and isinstance(result["risk_features"], dict)
        and result["risk_features"]
    ):
        html_parts.append("<h3>Risk Features</h3>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Feature</th><th>Coefficient</th></tr>")

        risk_features = result["risk_features"]
        # Sort risk features by coefficient value (descending)
        sorted_risk_features = sorted(
            risk_features.items(), key=lambda x: x[1], reverse=True
        )
        for feature, value in sorted_risk_features:
            html_parts.append(f"<tr><td>{feature}</td><td>{value:.6e}</td></tr>")

        html_parts.append("</table>")

    # Add contribution features table if available, sorted by coefficient descending
    if (
        "contribution_features" in result
        and isinstance(result["contribution_features"], dict)
        and result["contribution_features"]
    ):
        html_parts.append("<h3>Contribution Features</h3>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Feature</th><th>Coefficient</th></tr>")

        contribution_features = result["contribution_features"]
        # Sort contribution features by coefficient value (descending)
        sorted_contribution_features = sorted(
            contribution_features.items(), key=lambda x: x[1], reverse=True
        )
        for feature, value in sorted_contribution_features:
            html_parts.append(f"<tr><td>{feature}</td><td>{value:.6e}</td></tr>")

        html_parts.append("</table>")

    return "".join(html_parts)
