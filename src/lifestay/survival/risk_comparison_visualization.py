from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lifelines.fitters.coxph_fitter import CoxPHFitter as CoxPHModel
from plotly.subplots import make_subplots
from scipy import stats

from lifestay.survival.contribution_analysis import calculate_feature_contribution
from lifestay.survival.metrics_calculation import calculate_risk_metrics


class RiskComparisonVisualizer:
    """
    Class for creating interactive visualizations comparing different risk metrics
    from survival models for the same samples.

    This visualizer allows comparing the standard risk score distribution with
    feature contribution scores, presenting how they jointly influence outcomes.
    """

    def __init__(
        self,
        model: Optional[CoxPHModel] = None,
        focal_feature: Optional[str] = None,
        interaction_pattern: str = ":",
        colormap: str = "viridis",
    ):
        """
        Initialize the risk comparison visualizer.

        Args:
            model: A fitted Cox proportional hazards model
            focal_feature: The feature to analyze contributions for
            interaction_pattern: Pattern used to identify interaction terms
            colormap: Colormap to use for visualizations
        """
        self.model = model
        self.focal_feature = focal_feature
        self.interaction_pattern = interaction_pattern
        self.colormap = colormap
        self._analysis_df = None
        self._risk_scores = None
        self._contributions = None

    def with_model(self, model: CoxPHModel) -> "RiskComparisonVisualizer":
        """
        Set the model for analysis.

        Args:
            model: A fitted Cox proportional hazards model

        Returns:
            Self for method chaining
        """
        self.model = model
        return self

    def with_focal_feature(self, focal_feature: str) -> "RiskComparisonVisualizer":
        """
        Set the focal feature for contribution analysis.

        Args:
            focal_feature: The feature to analyze contributions for

        Returns:
            Self for method chaining
        """
        self.focal_feature = focal_feature
        return self

    def with_interaction_pattern(self, pattern: str) -> "RiskComparisonVisualizer":
        """
        Set the interaction pattern to identify interaction terms.

        Args:
            pattern: The pattern string (typically ":")

        Returns:
            Self for method chaining
        """
        self.interaction_pattern = pattern
        return self

    def with_colormap(self, colormap: str) -> "RiskComparisonVisualizer":
        """
        Set the colormap for visualizations.

        Args:
            colormap: Name of a valid colormap

        Returns:
            Self for method chaining
        """
        self.colormap = colormap
        return self

    def _calculate_metrics(self) -> None:
        """
        Calculate risk scores and feature contributions from the model for the same samples.

        This method calculates:
        1. Standard risk scores from the Cox model
        2. Feature contribution scores for the focal feature

        Both metrics are calculated for the same samples and stored in a DataFrame
        to ensure proper correspondence.
        """
        if self.model is None:
            raise ValueError("Model must be set before calculating metrics")

        # Use the extracted function to calculate metrics
        self._analysis_df = calculate_risk_metrics(
            model=self.model,
            focal_feature=self.focal_feature,
            interaction_pattern=self.interaction_pattern,
        )

        # Store individual metrics for convenience
        assert self._analysis_df is not None
        self._risk_scores = self._analysis_df["risk_score"]
        self._contributions = self._analysis_df["contribution"]

    def create_joint_scatter(
        self,
        height: int = 600,
        width: int = 800,
        opacity: float = 0.7,
        size: int = 8,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create an interactive scatter plot showing the relationship between
        risk scores and feature contributions for the same samples.

        Args:
            height (int): Height of the plot in pixels. Defaults to 600.
            width (int): Width of the plot in pixels. Defaults to 800.
            opacity (float): Opacity of the markers. Defaults to 0.7.
            size (int): Size of the markers. Defaults to 8.
            title (Optional[str]): Title for the plot. If None, a default title is generated.

        Returns:
            go.Figure: A plotly Figure object containing the visualization.
        """
        if self._analysis_df is None:
            self._calculate_metrics()

        if self._analysis_df is None:
            raise ValueError(
                "Failed to calculate metrics. Make sure the model is properly set."
            )

        # Determine title
        if title is None:
            focal_feature = (
                self.focal_feature if self.focal_feature else "unknown feature"
            )
            title = f"Risk Score vs. Contribution of '{focal_feature}' for Same Samples"

        # Create scatter plot with event status as color
        fig = px.scatter(
            self._analysis_df,
            x="risk_score",
            y="contribution",
            color="event",
            opacity=opacity,
            size_max=size,
            color_continuous_scale=self.colormap,
            title=title,
            labels={
                "risk_score": "Cox Risk Score",
                "contribution": f"Contribution from {self.focal_feature}",
                "event": "Event Status (0=censored, 1=observed)",
            },
            height=height,
            width=width,
        )

        # Add a trend line
        fig.add_trace(
            go.Scatter(
                x=self._analysis_df["risk_score"],
                y=np.poly1d(
                    np.polyfit(
                        self._analysis_df["risk_score"],
                        self._analysis_df["contribution"],
                        1,
                    )
                )(self._analysis_df["risk_score"]),
                mode="lines",
                name="Trend Line",
                line=dict(color="rgba(0,0,0,0.5)", width=2, dash="dash"),
            )
        )

        # Update layout
        fig.update_layout(
            xaxis_title="Cox Risk Score",
            yaxis_title=f"Contribution from {self.focal_feature}",
            legend_title="Event Status",
            margin=dict(l=50, r=50, b=50, t=50),
        )

        return fig

    def create_quadrant_analysis(
        self,
        height: int = 700,
        width: int = 900,
        title: Optional[str] = None,
        auto_scale: bool = True,
        opacity: float = 0.5,
        marker_size: int = 6,
        colormap: Optional[str] = None,
        color_discrete_map: Optional[Dict[int, str]] = None,
    ) -> go.Figure:
        """
        Create a quadrant analysis plot dividing samples into four groups based on
        their risk scores and feature contributions.

        This visualization helps understand how the two risk metrics jointly influence outcomes.

        Args:
            height (int): Height of the plot in pixels. Defaults to 700.
            width (int): Width of the plot in pixels. Defaults to 900.
            title (Optional[str]): Title for the plot. If None, a default title is generated.
            auto_scale (bool): Whether to automatically apply log scaling to highly
                skewed distributions. Defaults to True.
            opacity (float): Opacity of markers (0.0 to 1.0). Defaults to 0.5.
            marker_size (int): Size of markers. Defaults to 6.
            colormap (Optional[str]): Colormap to use for the plot. If None,
                uses the visualizer's default colormap.
            color_discrete_map (Optional[Dict[int, str]]): Discrete color mapping for event status.
                Defaults to None, which uses {0: "#2E5A88", 1: "#FFDD00"}.

        Returns:
            go.Figure: A plotly Figure object containing the quadrant analysis.
        """
        if self._analysis_df is None:
            self._calculate_metrics()

        if self._analysis_df is None:
            raise ValueError(
                "Failed to calculate metrics. Make sure the model is properly set."
            )

        # Use specified colormap or default
        colormap_to_use = colormap if colormap is not None else self.colormap

        # Default to blue and yellow color scheme for better visibility
        if color_discrete_map is None:
            color_discrete_map = {0: "#2E5A88", 1: "#FFDD00"}

        # Get data from the analysis DataFrame
        plot_df = self._analysis_df.copy()

        # Check for skewness and apply transformations if needed
        x_transform, y_transform = _determine_axis_transformations(
            plot_df["risk_score"], plot_df["contribution"], auto_scale
        )

        # Apply transformations
        plot_df["risk_score_transformed"] = x_transform(plot_df["risk_score"])
        plot_df["contribution_transformed"] = y_transform(plot_df["contribution"])

        # Get feature name - try to get it from the model if not set directly
        focal_feature_name = None

        # If focal_feature is already set, use it
        if self.focal_feature:
            focal_feature_name = self.focal_feature
        # Otherwise, try to extract from results summary if available
        elif self.model is not None:
            # Directly check if model has focal_feature attribute
            if hasattr(self.model, "focal_feature") and self.model.focal_feature:
                focal_feature_name = self.model.focal_feature

        # If still no focal_feature found, use default name
        if focal_feature_name is None:
            focal_feature_name = "unknown feature"

        # Get transformation names for axis labels
        x_label = _get_transform_label("Risk Score", x_transform)
        y_label = _get_transform_label(
            f"Contribution from {focal_feature_name}", y_transform
        )

        # Calculate median lines for quadrant division
        x_median = np.median(plot_df["risk_score_transformed"])
        y_median = np.median(plot_df["contribution_transformed"])

        # Create quadrant analysis plot
        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"type": "scatter"}]],
            subplot_titles=["Risk Score vs Feature Contribution Quadrant Analysis"],
        )

        # Get event status color mapping
        if isinstance(colormap_to_use, str):
            # If it's a string color mapping, use plotly's color mapping
            marker_color_spec = dict(
                color=plot_df["event"],
                colorscale=colormap_to_use,
                colorbar=dict(
                    title="Event Status",
                    tickvals=[0, 1],
                    ticktext=["No Event", "Event"],
                ),
            )
        else:
            # If it's a discrete color mapping, prepare color list
            marker_color_spec = dict(
                color=[
                    color_discrete_map.get(status, "#888888")
                    for status in plot_df["event"]
                ],
                showscale=False,
            )

        # Add scatter plot with improved visibility
        fig.add_trace(
            go.Scatter(
                x=plot_df["risk_score_transformed"],
                y=plot_df["contribution_transformed"],
                mode="markers",
                marker=dict(size=marker_size, opacity=opacity, **marker_color_spec),
                hovertemplate=(
                    "<b>Risk Score</b>: %{customdata[0]:.4f}<br>"
                    + f"<b>Contribution from {focal_feature_name}</b>: %{{customdata[1]:.4f}}<br>"
                    + "<b>Event</b>: %{customdata[2]}<br>"
                    + "<extra></extra>"
                ),
                customdata=np.column_stack(
                    (
                        plot_df["risk_score"],
                        plot_df["contribution"],
                        plot_df["event"].map({1: "Yes", 0: "No"}),
                    )
                ),
            )
        )

        # Add quadrant lines
        fig.add_hline(y=y_median, line=dict(color="gray", width=1, dash="dash"))
        fig.add_vline(x=x_median, line=dict(color="gray", width=1, dash="dash"))

        # Calculate statistics for each quadrant
        quadrant_stats = _calculate_quadrant_statistics(plot_df, x_median, y_median)

        # Add quadrant annotations
        add_quadrant_annotations(fig, x_median, y_median, quadrant_stats)

        # High risk quadrant highlight
        fig.add_shape(
            type="rect",
            x0=x_median,
            y0=y_median,
            x1=max(plot_df["risk_score_transformed"]) * 1.05,
            y1=max(plot_df["contribution_transformed"]) * 1.05,
            line=dict(color="red", width=2),
            fillcolor="red",
            opacity=0.05,
        )

        # Set layout
        default_title = f"Risk Quadrant Analysis: Cox Risk Score vs. Contribution from {focal_feature_name}"
        fig.update_layout(
            title=title if title else default_title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=height,
            width=width,
            hovermode="closest",
            template="plotly_white",
            plot_bgcolor="rgba(250, 250, 250, 0.9)",  # Use lighter background color
        )

        return fig

    def create_heatmap_analysis(
        self,
        bins: int = 10,
        height: int = 600,
        width: int = 800,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create a heatmap showing event rates across different combinations of
        risk scores and feature contributions.

        Args:
            bins (int): Number of bins for both axes. Defaults to 10.
            height (int): Height of the plot in pixels. Defaults to 600.
            width (int): Width of the plot in pixels. Defaults to 800.
            title (Optional[str]): Title for the plot. If None, a default title is generated.

        Returns:
            go.Figure: A plotly Figure object containing the heatmap visualization.
        """
        if self._analysis_df is None:
            self._calculate_metrics()

        if self._analysis_df is None:
            raise ValueError(
                "Failed to calculate metrics. Make sure the model is properly set."
            )

        # Determine title
        if title is None:
            focal_feature = (
                self.focal_feature if self.focal_feature else "unknown feature"
            )
            title = (
                f"Event Rate Heatmap: Risk Score vs. Contribution of '{focal_feature}'"
            )

        # Create bins for risk score and contribution
        risk_bins = pd.qcut(self._analysis_df["risk_score"], bins, duplicates="drop")
        contribution_bins = pd.qcut(
            self._analysis_df["contribution"], bins, duplicates="drop"
        )

        # Group by bins and calculate event rate
        heatmap_data = self._analysis_df.groupby([risk_bins, contribution_bins]).agg(
            {"event": ["count", "mean"], "time": "mean"}
        )
        heatmap_data.columns = ["count", "event_rate", "avg_time"]
        heatmap_data = heatmap_data.reset_index()

        # Convert bin categories to string labels for plotting
        heatmap_data["risk_bin"] = heatmap_data[risk_bins.name].apply(
            lambda x: f"{x.left:.2f}-{x.right:.2f}"
        )
        heatmap_data["contribution_bin"] = heatmap_data[contribution_bins.name].apply(
            lambda x: f"{x.left:.2f}-{x.right:.2f}"
        )

        # Create pivot table for heatmap
        pivot_data = heatmap_data.pivot_table(
            values="event_rate", index="contribution_bin", columns="risk_bin"
        )

        # Create heatmap
        fig = px.imshow(
            pivot_data,
            labels=dict(
                x="Risk Score Range", y="Contribution Range", color="Event Rate"
            ),
            x=pivot_data.columns,
            y=pivot_data.index,
            color_continuous_scale=self.colormap,
            title=title,
            height=height,
            width=width,
        )

        # Update layout
        fig.update_layout(
            xaxis_title="Cox Risk Score Range",
            yaxis_title=f"Contribution Range from {self.focal_feature}",
            coloraxis_colorbar=dict(title="Event Rate"),
            margin=dict(l=50, r=50, b=50, t=50),
        )

        return fig

    def create_3d_surface_analysis(
        self,
        bins: int = 30,
        height: int = 700,
        width: int = 900,
        title: Optional[str] = None,
        auto_scale: bool = True,
        opacity: float = 0.7,
        colormap: Optional[str] = None,
        marker_size: int = 6,
    ) -> go.Figure:
        """
        Create a 3D surface visualization showing risk score vs feature contribution
        with separate surfaces for event and non-event cases.

        Args:
            bins (int): Number of bins for creating the mesh grid. Defaults to 30.
            height (int): Height of the plot in pixels. Defaults to 700.
            width (int): Width of the plot in pixels. Defaults to 900.
            title (Optional[str]): Title for the plot. If None, a default title is generated.
            auto_scale (bool): Whether to automatically apply log scaling to highly
                skewed distributions. Defaults to True.
            opacity (float): Opacity of the surface. Defaults to 0.7.
            colormap (Optional[str]): Colormap to use for the plot. If None,
                uses the visualizer's default colormap.
            marker_size (int): Not used in this visualization, but included for compatibility
                with the visualize() method. Defaults to 6.

        Returns:
            go.Figure: A plotly Figure object containing the 3D surface visualization.
        """
        if self._analysis_df is None:
            self._calculate_metrics()

        if self._analysis_df is None:
            raise ValueError(
                "Failed to calculate metrics. Make sure the model is properly set."
            )

        # Use specified colormap or default
        colormap_to_use = colormap if colormap is not None else self.colormap

        # Get data from the analysis DataFrame
        plot_df = self._analysis_df.copy()

        # Check for skewness and apply transformations if needed
        x_transform, y_transform = _determine_axis_transformations(
            plot_df["risk_score"], plot_df["contribution"], auto_scale
        )

        # Apply transformations
        plot_df["risk_score_transformed"] = x_transform(plot_df["risk_score"])
        plot_df["contribution_transformed"] = y_transform(plot_df["contribution"])

        # Get feature name
        focal_feature_name = None
        if self.focal_feature:
            focal_feature_name = self.focal_feature
        elif (
            self.model is not None
            and hasattr(self.model, "focal_feature")
            and self.model.focal_feature
        ):
            focal_feature_name = self.model.focal_feature
        if focal_feature_name is None:
            focal_feature_name = "unknown feature"

        # Get transformation names for axis labels
        x_label = _get_transform_label("Risk Score", x_transform)
        y_label = _get_transform_label(
            f"Contribution from {focal_feature_name}", y_transform
        )

        # Create separate dataframes for event and non-event cases
        event_df = plot_df[plot_df["event"] == 1]
        non_event_df = plot_df[plot_df["event"] == 0]

        # Create mesh grid for both event types
        def create_mesh_data(df, density_scale=1.0):
            # Create linearly spaced coordinates
            x_range = np.linspace(
                df["risk_score_transformed"].min(),
                df["risk_score_transformed"].max(),
                bins,
            )
            y_range = np.linspace(
                df["contribution_transformed"].min(),
                df["contribution_transformed"].max(),
                bins,
            )

            # Create mesh grid
            x_mesh, y_mesh = np.meshgrid(x_range, y_range)

            # Initialize density grid
            density_grid = np.zeros((bins, bins))

            # Calculate 2D histogram for density
            hist, x_edges, y_edges = np.histogram2d(
                df["risk_score_transformed"],
                df["contribution_transformed"],
                bins=[x_range, y_range],
            )

            # Normalize and scale density
            density_grid = hist.T * density_scale

            # Apply logarithmic scaling to density for better visualization
            density_grid = np.log1p(density_grid)  # log(1+x) to handle zeros

            return x_mesh, y_mesh, density_grid

        # Get common x and y ranges for both surfaces to ensure alignment
        # This ensures both surfaces use the same coordinate space
        x_min = min(
            plot_df["risk_score_transformed"].min(),
            plot_df["risk_score_transformed"].min(),
        )
        x_max = max(
            plot_df["risk_score_transformed"].max(),
            plot_df["risk_score_transformed"].max(),
        )
        y_min = min(
            plot_df["contribution_transformed"].min(),
            plot_df["contribution_transformed"].min(),
        )
        y_max = max(
            plot_df["contribution_transformed"].max(),
            plot_df["contribution_transformed"].max(),
        )

        # Create common grid ranges
        x_range = np.linspace(x_min, x_max, bins)
        y_range = np.linspace(y_min, y_max, bins)

        # Define alternative create_mesh_data that uses the common ranges
        def create_mesh_data_common_grid(df, density_scale=1.0):
            # Create mesh grid
            x_mesh, y_mesh = np.meshgrid(x_range, y_range)

            # Calculate 2D histogram for density
            hist, _, _ = np.histogram2d(
                df["risk_score_transformed"],
                df["contribution_transformed"],
                bins=[x_range, y_range],
            )

            # Normalize and scale density
            density_grid = hist.T * density_scale

            # Apply logarithmic scaling to density for better visualization
            density_grid = np.log1p(density_grid)  # log(1+x) to handle zeros

            return x_mesh, y_mesh, density_grid

        # Calculate density scaling based on data size
        event_scale = 1.0
        non_event_scale = 1.0

        # If one group is significantly smaller, scale its density to make it more visible
        if len(event_df) < len(non_event_df) * 0.2:
            event_scale = 2.0
        elif len(non_event_df) < len(event_df) * 0.2:
            non_event_scale = 2.0

        # Create mesh data for event and non-event cases using common grid
        x_event, y_event, z_event = create_mesh_data_common_grid(event_df, event_scale)
        x_non_event, y_non_event, z_non_event = create_mesh_data_common_grid(
            non_event_df, non_event_scale
        )

        # Offset one surface slightly for better visibility when overlapping
        z_offset = 0
        if z_event.max() > 0 and z_non_event.max() > 0:
            # Add a small offset to event surface for better visibility
            z_offset = min(z_event.max(), z_non_event.max()) * 0.05

        # Set up 3D figure
        fig = go.Figure()

        # Add surface for non-event cases
        fig.add_trace(
            go.Surface(
                x=x_non_event,
                y=y_non_event,
                z=z_non_event,
                colorscale="Blues",  # Blue colorscale for non-events
                opacity=opacity,
                showscale=True,
                name="No Event",
                hovertemplate="<b>Risk Score</b>: %{x:.4f}<br><b>Contribution</b>: %{y:.4f}<br><b>Density</b>: %{z:.4f}<extra>No Event</extra>",
                colorbar=dict(
                    title="No Event Density",
                    x=1.0,
                ),
                contours=dict(
                    x=dict(show=True, width=1, color="rgba(150,150,150,0.5)"),
                    y=dict(show=True, width=1, color="rgba(150,150,150,0.5)"),
                    z=dict(show=True, width=1, color="rgba(150,150,150,0.5)"),
                ),
                lighting=dict(
                    ambient=0.6,
                    diffuse=0.8,
                    roughness=0.4,
                    specular=0.3,
                ),
            )
        )

        # Add surface for event cases (with slight z-offset for better visibility)
        fig.add_trace(
            go.Surface(
                x=x_event,
                y=y_event,
                z=z_event + z_offset,  # Add small offset for better visibility
                colorscale="YlOrRd",  # Yellow to red colorscale for events
                opacity=opacity,
                showscale=True,
                name="Event",
                hovertemplate="<b>Risk Score</b>: %{x:.4f}<br><b>Contribution</b>: %{y:.4f}<br><b>Density</b>: %{z:.4f}<extra>Event</extra>",
                colorbar=dict(
                    title="Event Density",
                    x=0.85,
                ),
                contours=dict(
                    x=dict(show=True, width=1, color="rgba(150,150,150,0.5)"),
                    y=dict(show=True, width=1, color="rgba(150,150,150,0.5)"),
                    z=dict(show=True, width=1, color="rgba(150,150,150,0.5)"),
                ),
                lighting=dict(
                    ambient=0.6,
                    diffuse=0.8,
                    roughness=0.4,
                    specular=0.3,
                ),
            )
        )

        # Add text annotations for event rates
        event_rate = len(event_df) / len(plot_df) * 100 if len(plot_df) > 0 else 0
        non_event_rate = (
            len(non_event_df) / len(plot_df) * 100 if len(plot_df) > 0 else 0
        )

        # Create annotations
        annotations = [
            dict(
                showarrow=False,
                x=0.85,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"<b>Event Rate</b>: {event_rate:.1f}%<br><b>N</b>: {len(event_df)}",
                font=dict(size=12, color="#C25450"),
                align="left",
            ),
            dict(
                showarrow=False,
                x=0.85,
                y=0.9,
                xref="paper",
                yref="paper",
                text=f"<b>No Event Rate</b>: {non_event_rate:.1f}%<br><b>N</b>: {len(non_event_df)}",
                font=dict(size=12, color="#3366CC"),
                align="left",
            ),
        ]

        # Set layout with better camera angle
        default_title = f"Risk Score vs Feature Contribution: 3D Density Analysis"
        fig.update_layout(
            title=title if title else default_title,
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title="Density (log scale)",
                xaxis=dict(gridcolor="rgb(230, 230, 230)"),
                yaxis=dict(gridcolor="rgb(230, 230, 230)"),
                zaxis=dict(gridcolor="rgb(230, 230, 230)"),
                # Set camera angle for better initial view
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0),
                ),
                aspectratio=dict(
                    x=1, y=1, z=0.7
                ),  # Adjust aspect ratio for better visibility
            ),
            height=height,
            width=width,
            margin=dict(l=65, r=65, b=65, t=90),
            template="plotly_white",
            annotations=annotations,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        return fig

    def create_interactive_dashboard(
        self,
        height: int = 1200,
        width: int = 1200,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create a comprehensive interactive dashboard with multiple visualization types
        showing the relationship between risk scores and feature contributions.

        This dashboard includes:
        1. Joint scatter plot of risk vs contribution colored by event status
        2. Quadrant analysis dividing samples into risk groups
        3. Heatmap showing event rates across different risk combinations
        4. Histograms for both risk scores and contributions

        Args:
            height (int): Height of the dashboard in pixels. Defaults to 1200.
            width (int): Width of the dashboard in pixels. Defaults to 1200.
            title (Optional[str]): Title for the dashboard. If None, a default title is generated.

        Returns:
            go.Figure: A plotly Figure object containing the interactive dashboard.
        """
        if self._analysis_df is None:
            self._calculate_metrics()

        if self._analysis_df is None:
            raise ValueError(
                "Failed to calculate metrics. Make sure the model is properly set."
            )

        # Determine title
        if title is None:
            focal_feature = (
                self.focal_feature if self.focal_feature else "unknown feature"
            )
            title = f"Joint Risk Analysis Dashboard for '{focal_feature}'"

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
            subplot_titles=(
                "Risk vs Contribution Scatter",
                "Quadrant Analysis",
                "Event Rate Heatmap",
                "Risk Distributions",
            ),
        )

        # 1. Joint scatter plot (top left)
        fig.add_trace(
            go.Scatter(
                x=self._analysis_df["risk_score"],
                y=self._analysis_df["contribution"],
                mode="markers",
                marker=dict(
                    color=self._analysis_df["event"],
                    colorscale=self.colormap,
                    opacity=0.7,
                    showscale=True,
                    colorbar=dict(title="Event Status"),
                ),
                name="Samples",
            ),
            row=1,
            col=1,
        )

        # Add trend line
        fig.add_trace(
            go.Scatter(
                x=self._analysis_df["risk_score"],
                y=np.poly1d(
                    np.polyfit(
                        self._analysis_df["risk_score"],
                        self._analysis_df["contribution"],
                        1,
                    )
                )(self._analysis_df["risk_score"]),
                mode="lines",
                name="Trend",
                line=dict(color="rgba(0,0,0,0.5)", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

        # 2. Quadrant analysis (top right)
        # Calculate medians for dividing into quadrants
        risk_median = self._analysis_df["risk_score"].median()
        contribution_median = self._analysis_df["contribution"].median()

        # Create quadrant labels
        self._analysis_df["quadrant"] = "Q0"
        self._analysis_df.loc[
            (self._analysis_df["risk_score"] >= risk_median)
            & (self._analysis_df["contribution"] >= contribution_median),
            "quadrant",
        ] = "Q1: High Risk, High Contribution"
        self._analysis_df.loc[
            (self._analysis_df["risk_score"] < risk_median)
            & (self._analysis_df["contribution"] >= contribution_median),
            "quadrant",
        ] = "Q2: Low Risk, High Contribution"
        self._analysis_df.loc[
            (self._analysis_df["risk_score"] < risk_median)
            & (self._analysis_df["contribution"] < contribution_median),
            "quadrant",
        ] = "Q3: Low Risk, Low Contribution"
        self._analysis_df.loc[
            (self._analysis_df["risk_score"] >= risk_median)
            & (self._analysis_df["contribution"] < contribution_median),
            "quadrant",
        ] = "Q4: High Risk, Low Contribution"

        # Add scatter plot with quadrant coloring
        for quadrant in [
            "Q1: High Risk, High Contribution",
            "Q2: Low Risk, High Contribution",
            "Q3: Low Risk, Low Contribution",
            "Q4: High Risk, Low Contribution",
        ]:
            mask = self._analysis_df["quadrant"] == quadrant
            if mask.sum() > 0:
                fig.add_trace(
                    go.Scatter(
                        x=self._analysis_df.loc[mask, "risk_score"],
                        y=self._analysis_df.loc[mask, "contribution"],
                        mode="markers",
                        name=quadrant,
                        marker=dict(opacity=0.7),
                    ),
                    row=1,
                    col=2,
                )

        # Add quadrant dividing lines
        fig.add_shape(
            type="line",
            line=dict(dash="dash", width=1, color="gray"),
            x0=risk_median,
            y0=self._analysis_df["contribution"].min(),
            x1=risk_median,
            y1=self._analysis_df["contribution"].max(),
            row=1,
            col=2,
        )
        fig.add_shape(
            type="line",
            line=dict(dash="dash", width=1, color="gray"),
            x0=self._analysis_df["risk_score"].min(),
            y0=contribution_median,
            x1=self._analysis_df["risk_score"].max(),
            y1=contribution_median,
            row=1,
            col=2,
        )

        # 3. Heatmap (bottom left)
        # Create bins for risk score and contribution
        bins = 8  # Number of bins for heatmap
        risk_bins = pd.qcut(self._analysis_df["risk_score"], bins, duplicates="drop")
        contribution_bins = pd.qcut(
            self._analysis_df["contribution"], bins, duplicates="drop"
        )

        # Group by bins and calculate event rate
        heatmap_data = (
            self._analysis_df.groupby([risk_bins, contribution_bins])["event"]
            .mean()
            .reset_index()
        )

        # Convert bin categories to numeric values for plotting
        heatmap_data["risk_bin_num"] = heatmap_data[risk_bins.name].cat.codes
        heatmap_data["contribution_bin_num"] = heatmap_data[
            contribution_bins.name
        ].cat.codes

        # Create pivot table for heatmap
        pivot_data = heatmap_data.pivot_table(
            values="event", index="contribution_bin_num", columns="risk_bin_num"
        )

        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=pivot_data.values,
                colorscale=self.colormap,
                showscale=True,
                colorbar=dict(title="Event Rate"),
            ),
            row=2,
            col=1,
        )

        # 4. Risk distributions (bottom right)
        # Add risk score histogram
        fig.add_trace(
            go.Histogram(
                x=self._analysis_df["risk_score"],
                name="Risk Score",
                marker_color="rgba(73, 160, 235, 0.7)",
                opacity=0.7,
                nbinsx=20,
            ),
            row=2,
            col=2,
        )

        # Add contribution histogram
        fig.add_trace(
            go.Histogram(
                x=self._analysis_df["contribution"],
                name="Feature Contribution",
                marker_color="rgba(235, 73, 160, 0.7)",
                opacity=0.7,
                nbinsx=20,
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            width=width,
            showlegend=True,
        )

        # Update axes titles
        fig.update_xaxes(title_text="Cox Risk Score", row=1, col=1)
        fig.update_yaxes(
            title_text=f"Contribution from {self.focal_feature}", row=1, col=1
        )

        fig.update_xaxes(title_text="Cox Risk Score", row=1, col=2)
        fig.update_yaxes(
            title_text=f"Contribution from {self.focal_feature}", row=1, col=2
        )

        fig.update_xaxes(title_text="Risk Score Bin", row=2, col=1)
        fig.update_yaxes(title_text="Contribution Bin", row=2, col=1)

        fig.update_xaxes(title_text="Value", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)

        return fig

    def visualize(
        self,
        plot_type: str = "dashboard",
        auto_scale: bool = True,
        opacity: float = 0.5,
        marker_size: int = 6,
        colormap: Optional[str] = None,
        color_discrete_map: Optional[Dict[int, str]] = None,
        **kwargs: Any,
    ) -> go.Figure:
        """
        Create and display a visualization based on the specified type.

        Args:
            plot_type (str): Type of plot to create. Options are:
                - "scatter": Joint scatter plot of risk vs contribution
                - "quadrant": Quadrant analysis dividing samples into risk groups
                - "heatmap": Heatmap showing event rates across different risk combinations
                - "3d_surface": 3D surface visualization showing risk score vs feature contribution
                - "dashboard": Interactive dashboard with multiple plots
            auto_scale (bool): Whether to automatically apply log scaling to highly
                skewed distributions. Defaults to True.
            opacity (float): Opacity of markers (0.0 to 1.0). Defaults to 0.5.
            marker_size (int): Size of markers. Defaults to 6.
            colormap (Optional[str]): Override the default colormap.
            color_discrete_map (Optional[Dict[int, str]]): Discrete color mapping for event status.
                Example: {0: "#2E5A88", 1: "#FFDD00"} for no-event and event.
            **kwargs: Additional arguments to pass to the specific plotting function.

        Returns:
            go.Figure: The generated plotly figure.

        Raises:
            ValueError: If an invalid plot_type is specified.
        """
        if self.model is None:
            raise ValueError("Model must be set before visualization")

        # Calculate metrics if not already done
        if self._analysis_df is None:
            self._calculate_metrics()

        # Pass parameters to kwargs if not already present
        if "auto_scale" not in kwargs:
            kwargs["auto_scale"] = auto_scale
        if "opacity" not in kwargs:
            kwargs["opacity"] = opacity
        if "marker_size" not in kwargs:
            kwargs["marker_size"] = marker_size
        if colormap is not None and "colormap" not in kwargs:
            kwargs["colormap"] = colormap
        if color_discrete_map is not None and "color_discrete_map" not in kwargs:
            kwargs["color_discrete_map"] = color_discrete_map

        # Create the appropriate visualization
        if plot_type == "scatter":
            return self.create_joint_scatter(**kwargs)
        elif plot_type == "quadrant":
            return self.create_quadrant_analysis(**kwargs)
        elif plot_type == "heatmap":
            return self.create_heatmap_analysis(**kwargs)
        elif plot_type == "3d_surface":
            return self.create_3d_surface_analysis(**kwargs)
        elif plot_type == "dashboard":
            return self.create_interactive_dashboard(**kwargs)
        else:
            raise ValueError(
                f"Invalid plot_type: {plot_type}. Valid options are: "
                "'scatter', 'quadrant', 'heatmap', '3d_surface', 'dashboard'"
            )


def visualize_risk_quadrants(
    model: CoxPHModel,
    focal_feature: Optional[str] = None,
    interaction_pattern: str = ":",
    auto_scale: bool = True,
    output_path: str = "risk_quadrants.html",
    width: int = 900,
    height: int = 700,
    title: Optional[str] = None,
    colormap: str = "viridis",
    return_fig: bool = False,
) -> Optional[go.Figure]:
    """
    Create an interactive quadrant analysis visualization comparing standard risk scores
    with feature contribution scores for the same samples.

    Args:
        model (CoxPHModel): The fitted Cox model
        focal_feature (Optional[str]): Feature to calculate contribution for.
                                      If None, uses model's focal_feature.
        interaction_pattern (str): Pattern used to identify interaction terms.
        auto_scale (bool): Whether to automatically apply log scaling to highly skewed distributions.
        output_path (str): Path to save the HTML visualization (default: "risk_quadrants.html").
        width (int): Width of the plot in pixels.
        height (int): Height of the plot in pixels.
        title (Optional[str]): Custom title for the plot. If None, a default title is generated.
        colormap (str): Colormap to use for the plot.
        return_fig (bool): Whether to return the figure object.

    Returns:
        Optional[go.Figure]: The plotly figure object if return_fig is True, else None.
    """
    # Get data from model
    X = model.X
    E = model.E

    # If focal_feature not provided, try to get from model
    if focal_feature is None:
        if hasattr(model, "focal_feature"):
            focal_feature = model.focal_feature
        else:
            raise ValueError("Focal feature must be provided or available in the model")

    # Ensure the focal feature is in the correct format (with or without _avg suffix)
    focal_feature_col = focal_feature
    if focal_feature not in X.columns and f"{focal_feature}_avg" in X.columns:
        focal_feature_col = f"{focal_feature}_avg"
        print(f"Using transformed focal feature name: {focal_feature_col}")

    # Calculate risk scores and feature contributions for each sample
    risk_scores = model.predict_risk(X)
    contributions = calculate_feature_contribution(
        model=model,
        X=X,
        focal_feature=focal_feature_col,
        interaction_pattern=interaction_pattern,
    )

    # Prepare data for plotting
    plot_df = pd.DataFrame(
        {
            "risk_score": risk_scores,
            "contribution": contributions,
            "event": E.astype(int),
        }
    )

    # Check for skewness and apply transformations if needed
    x_transform, y_transform = _determine_axis_transformations(
        risk_scores, contributions, auto_scale
    )

    # Apply transformations
    plot_df["risk_score_transformed"] = x_transform(plot_df["risk_score"])
    plot_df["contribution_transformed"] = y_transform(plot_df["contribution"])

    # Get transformation names for axis labels
    x_label = _get_transform_label("Risk Score", x_transform)
    y_label = _get_transform_label(f"Contribution from {focal_feature}", y_transform)

    # Calculate median lines for quadrant division
    x_median = np.median(plot_df["risk_score_transformed"])
    y_median = np.median(plot_df["contribution_transformed"])

    # Create quadrant analysis plot
    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"type": "scatter"}]],
        subplot_titles=["Risk Score vs Feature Contribution Quadrant Analysis"],
    )

    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=plot_df["risk_score_transformed"],
            y=plot_df["contribution_transformed"],
            mode="markers",
            marker=dict(
                size=8,
                color=plot_df["event"],
                colorscale=colormap,
                colorbar=dict(
                    title="Event Status",
                    tickvals=[0, 1],
                    ticktext=["No Event", "Event"],
                ),
                opacity=0.7,
            ),
            text=[
                f"Risk Score: {r:.4f}<br>"
                f"Contribution: {c:.4f}<br>"
                f"Event: {'Yes' if e else 'No'}"
                for r, c, e in zip(
                    plot_df["risk_score"], plot_df["contribution"], plot_df["event"]
                )
            ],
            hoverinfo="text",
        )
    )

    # Add quadrant dividing lines
    fig.add_shape(
        type="line",
        x0=x_median,
        y0=plot_df["contribution_transformed"].min(),
        x1=x_median,
        y1=plot_df["contribution_transformed"].max(),
        line=dict(color="black", width=1, dash="dash"),
    )

    fig.add_shape(
        type="line",
        x0=plot_df["risk_score_transformed"].min(),
        y0=y_median,
        x1=plot_df["risk_score_transformed"].max(),
        y1=y_median,
        line=dict(color="black", width=1, dash="dash"),
    )

    # Label the quadrants
    quadrants = [
        ["Low Risk<br>Low Contribution", "Low Risk<br>High Contribution"],
        ["High Risk<br>Low Contribution", "High Risk<br>High Contribution"],
    ]

    x_points = [
        plot_df["risk_score_transformed"].min()
        + 0.1 * (x_median - plot_df["risk_score_transformed"].min()),
        x_median + 0.1 * (plot_df["risk_score_transformed"].max() - x_median),
    ]

    y_points = [
        y_median + 0.1 * (plot_df["contribution_transformed"].max() - y_median),
        plot_df["contribution_transformed"].min()
        + 0.1 * (y_median - plot_df["contribution_transformed"].min()),
    ]

    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=x_points[i],
                y=y_points[j],
                text=quadrants[i][j],
                showarrow=False,
                font=dict(size=12, color="black", family="Arial"),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                align="center",
            )

    # Calculate statistics for each quadrant
    quadrant_stats = _calculate_quadrant_statistics(plot_df, x_median, y_median)

    # Add quadrant statistics to the plot
    stats_text = "<br>".join(
        [
            f"<b>Quadrant {i+1} ({q['label']}):</b> {q['count']} samples, "
            f"{q['event_count']} events ({q['event_rate']:.1%} event rate)"
            for i, q in enumerate(quadrant_stats)
        ]
    )

    fig.add_annotation(
        x=0.98,
        y=0.02,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        font=dict(size=10),
        align="right",
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
    )

    # Set title
    if title is None:
        title = f"Risk Score vs Feature Contribution: {focal_feature}"

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=width,
        height=height,
        template="plotly_white",
        legend_title="Event Status",
        hovermode="closest",
        margin=dict(l=60, r=30, t=50, b=60),
    )

    # Save to HTML file
    fig.write_html(output_path)
    print(f"Visualization saved to {output_path}")

    return fig if return_fig else None


def _determine_axis_transformations(
    risk_scores: pd.Series, contributions: pd.Series, auto_scale: bool = True
) -> Tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    """
    Determine appropriate transformations for risk score and contribution axes.

    Args:
        risk_scores (pd.Series): Risk scores
        contributions (pd.Series): Feature contributions
        auto_scale (bool): Whether to automatically apply transformations

    Returns:
        Tuple[Callable[[Any], Any], Callable[[Any], Any]]: Transformation functions for x and y axes
    """
    # Default is identity transformation (no change)
    identity = lambda x: x

    if not auto_scale:
        return identity, identity

    # Helper function to detect if transformation is needed
    def needs_transform(series: pd.Series) -> bool:
        # Calculate skewness
        skewness = stats.skew(series)

        # Calculate percentile ratios
        p99 = np.percentile(series, 99)
        p50 = np.percentile(series, 50)
        p1 = np.percentile(series, 1)

        # Handle potential zeros in denominators
        p99_p50_ratio = p99 / p50 if p50 != 0 else float("inf")
        p50_p1_ratio = p50 / p1 if p1 != 0 else float("inf")

        # Check for conditions that indicate need for transformation
        high_skew = abs(skewness) > 2.0
        percentile_imbalance = (p99_p50_ratio > 3.0) or (p50_p1_ratio > 3.0)

        return bool(high_skew or percentile_imbalance)

    # Choose transformations based on distribution properties
    x_transform = np.log1p if needs_transform(risk_scores) else identity
    y_transform = np.log1p if needs_transform(contributions) else identity

    return x_transform, y_transform


def _get_transform_label(base_label: str, transform_func: Callable[[Any], Any]) -> str:
    """
    Get appropriate axis label based on the transformation function.

    Args:
        base_label (str): Original axis label
        transform_func (Callable[[Any], Any]): Transformation function

    Returns:
        str: Updated axis label
    """
    if transform_func == np.log1p:
        return f"log(1 + {base_label})"
    elif transform_func == np.log:
        return f"log({base_label})"
    elif transform_func == np.sqrt:
        return f"sqrt({base_label})"
    else:
        return base_label


def _calculate_quadrant_statistics(
    plot_df: pd.DataFrame, x_median: float, y_median: float
) -> List[Dict[str, Any]]:
    """
    Calculate statistics for each quadrant in the plot.

    Args:
        plot_df (pd.DataFrame): Dataframe with risk scores, contributions and event status
        x_median (float): Median value for x-axis (risk score)
        y_median (float): Median value for y-axis (contribution)

    Returns:
        List[Dict[str, Any]]: List of dictionaries with quadrant statistics
    """
    # Create quadrant labels
    plot_df["quadrant"] = "Q0"  # Default

    # Q1: High Risk, High Contribution
    plot_df.loc[
        (plot_df["risk_score_transformed"] >= x_median)
        & (plot_df["contribution_transformed"] >= y_median),
        "quadrant",
    ] = "Q1: High Risk, High Contribution"

    # Q2: Low Risk, High Contribution
    plot_df.loc[
        (plot_df["risk_score_transformed"] < x_median)
        & (plot_df["contribution_transformed"] >= y_median),
        "quadrant",
    ] = "Q2: Low Risk, High Contribution"

    # Q3: Low Risk, Low Contribution
    plot_df.loc[
        (plot_df["risk_score_transformed"] < x_median)
        & (plot_df["contribution_transformed"] < y_median),
        "quadrant",
    ] = "Q3: Low Risk, Low Contribution"

    # Q4: High Risk, Low Contribution
    plot_df.loc[
        (plot_df["risk_score_transformed"] >= x_median)
        & (plot_df["contribution_transformed"] < y_median),
        "quadrant",
    ] = "Q4: High Risk, Low Contribution"

    # Calculate statistics for each quadrant
    quadrant_stats = []

    for quadrant, group in plot_df.groupby("quadrant"):
        if quadrant == "Q0":  # Skip default
            continue

        stats = {
            "quadrant": quadrant,
            "count": len(group),
            "event_rate": group["event"].mean() * 100,  # Convert to percentage
            "avg_time": group["time"].mean() if "time" in group.columns else None,
        }
        quadrant_stats.append(stats)

    return quadrant_stats


def add_quadrant_annotations(
    fig: go.Figure,
    x_median: float,
    y_median: float,
    quadrant_stats: List[Dict[str, Any]],
) -> None:
    """
    Add annotations to the quadrant plot with statistics.

    Args:
        fig (go.Figure): Plotly figure to add annotations to
        x_median (float): Median value for x-axis
        y_median (float): Median value for y-axis
        quadrant_stats (List[Dict[str, Any]]): List of dictionaries with quadrant statistics
    """
    annotations = []

    # Get axis limits from figure
    x_range = fig.layout.xaxis.range
    y_range = fig.layout.yaxis.range

    # If ranges not set yet, estimate from data
    if x_range is None or y_range is None:
        # Get traces data
        x_data = []
        y_data = []
        for trace in fig.data:
            if trace.x is not None and trace.y is not None:
                x_data.extend(trace.x)
                y_data.extend(trace.y)

        if x_data and y_data:
            x_min, x_max = min(x_data), max(x_data)
            y_min, y_max = min(y_data), max(y_data)
        else:
            # Fallback if no data found
            x_min, x_max = x_median * 0.5, x_median * 1.5
            y_min, y_max = y_median * 0.5, y_median * 1.5
    else:
        x_min, x_max = x_range[0], x_range[1]
        y_min, y_max = y_range[0], y_range[1]

    # Calculate positions for annotations
    positions = {
        "Q1: High Risk, High Contribution": {
            "x": x_median + (x_max - x_median) * 0.5,
            "y": y_median + (y_max - y_median) * 0.5,
        },
        "Q2: Low Risk, High Contribution": {
            "x": x_min + (x_median - x_min) * 0.5,
            "y": y_median + (y_max - y_median) * 0.5,
        },
        "Q3: Low Risk, Low Contribution": {
            "x": x_min + (x_median - x_min) * 0.5,
            "y": y_min + (y_median - y_min) * 0.5,
        },
        "Q4: High Risk, Low Contribution": {
            "x": x_median + (x_max - x_median) * 0.5,
            "y": y_min + (y_median - y_min) * 0.5,
        },
    }

    # Create annotations
    for stat in quadrant_stats:
        quadrant = stat["quadrant"]
        if quadrant not in positions:
            continue

        pos = positions[quadrant]

        # Build annotation text
        text = f"N: {stat['count']}<br>Event Rate: {stat['event_rate']:.1f}%"
        if stat["avg_time"] is not None:
            text += f"<br>Avg Time: {stat['avg_time']:.1f}"

        annotations.append(
            dict(x=pos["x"], y=pos["y"], text=text, showarrow=False, font=dict(size=10))
        )

    # Add annotations to figure
    fig.update_layout(annotations=annotations)
