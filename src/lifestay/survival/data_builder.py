"""
Survival Dataset Builder Module

This module provides a class-based approach to building survival analysis datasets
from SampleBuilder data, maintaining a functional programming style.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from ..loader.sample_builder import SampleBuilder
from .basic_cox import basic_cox_model
from .cox_model import CoxPHModel
from .data_converter import convert_to_lifelines_format
from .interaction_cox import cox_model_with_interactions


class SurvivalDatasetBuilder:
    """
    Builder class for creating survival datasets from CSV files.

    This class handles the entire pipeline for preparing survival analysis data:
    1. Loading and processing CSV data using SampleBuilder
    2. Converting processed data to lifelines format (X, T, E)
    3. Applying any necessary pre- or post-processing steps
    4. Filtering records with time-to-event less than the window size (by default)

    Usage example:
        builder = SurvivalDatasetBuilder() \
            .with_file("data.csv") \
            .with_feature_columns(["col1", "col2"]) \
            .with_heartbeat_column("heartbeat") \
            .with_patterns({"positive": [1, 1, 1], "negative": [0, 0, 0]}) \
            .with_filter_short_events(True) \
            .with_debug(True)

        X, T, E = builder.build()

    Attributes:
        file_path: Path to the input CSV file
        feature_columns: List of column names to calculate window averages for
                         (can be set later with with_feature_columns method)
        heartbeat_column: Column name for heartbeat data
                         (can be set later with with_heartbeat_column method)
        patterns: Dictionary mapping pattern names to pattern definitions
                 (can be set later with with_patterns method)
        window_size: Number of values to include in each average window
        event_value: Value in sample_type column that represents the event
        time_column: Column name to use for time-to-event
        time_from_window: Whether to use window indices to generate time values
        debug: Whether to print debug information
        filter_short_events: Whether to filter out records where time_to_event < window_size
    """

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        heartbeat_column: Optional[str] = None,
        patterns: Optional[Dict[str, List[int]]] = None,
        window_size: int = 5,
        event_value: str = "negative",
        time_column: Optional[str] = None,
        time_from_window: bool = True,
        debug: bool = False,
        filter_short_events: bool = True,
        focal_feature: Optional[str] = None,
    ):
        """
        Initialize the SurvivalDatasetBuilder.

        Args:
            feature_columns: List of column names to calculate window averages for
                         and to use as features in the survival model
            heartbeat_column: Column name to use for EOL pattern detection
            patterns: Dict mapping pattern names to pattern definitions
            window_size: Number of values to include in each average window
            event_value: Value in sample_type column that represents the event (will be converted to event=1)
            time_column: Column to use as time-to-event
            time_from_window: Whether to generate time values based on window distance to event
            debug: Whether to print debug information
            filter_short_events: Whether to filter out records where time_to_event < window_size
            focal_feature: Feature to interact with other features (for interaction models). Default is None.
        """
        self.file_path: Optional[str] = None
        self.feature_columns: Optional[List[str]] = feature_columns
        self.heartbeat_column: Optional[str] = heartbeat_column
        self.patterns: Optional[Dict[str, List[int]]] = patterns
        self.window_size: int = window_size
        self.event_value: str = event_value
        self.time_column: Optional[str] = time_column
        self.time_from_window: bool = time_from_window
        self.debug: bool = debug
        self.filter_short_events: bool = filter_short_events
        self.focal_feature: Optional[str] = focal_feature

        # Sample builder instance
        self.sample_builder: Optional[SampleBuilder] = None

        # Cache for available columns
        self._available_columns: Optional[List[str]] = None

        # Data containers
        self.combined_df: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.T: Optional[pd.Series] = None
        self.E: Optional[pd.Series] = None

        # Sample size for testing
        self.sample_size: Optional[int] = None

        # Hooks for pre and post processing
        self.pre_process_hooks: List[Callable[[pd.DataFrame], pd.DataFrame]] = []
        self.post_process_hooks: List[
            Callable[
                [pd.DataFrame, pd.Series, pd.Series],
                Tuple[pd.DataFrame, pd.Series, pd.Series],
            ]
        ] = []

    def __call__(self, file_path: str, sample_size: Optional[int] = None) -> CoxPHModel:
        """
        Build the survival dataset when the instance is called as a function.

        Args:
            file_path: Path to the CSV file to process
            sample_size: Optional number of rows to process

        Returns:
            CoxPHModel: Fitted Cox Proportional Hazards model
        """
        if self.file_path is None or self.file_path != file_path:
            self.file_path = file_path

        self.sample_size = sample_size
        return self.build()

    def with_file(self, file_path: str) -> "SurvivalDatasetBuilder":
        """
        Set the input file path for the dataset.

        Args:
            file_path: Path to the CSV file to process

        Returns:
            Self for method chaining
        """
        self.file_path = file_path
        self._initialize_sample_builder()
        return self

    def with_feature_columns(
        self, feature_columns: List[str]
    ) -> "SurvivalDatasetBuilder":
        """
        Set the feature columns to use.

        Args:
            feature_columns: List of column names to calculate window averages for
                        and to use as features in the survival model

        Returns:
            Self for method chaining
        """
        self.feature_columns = feature_columns
        return self

    def with_heartbeat_column(self, heartbeat_column: str) -> "SurvivalDatasetBuilder":
        """
        Set the heartbeat column for EOL pattern detection.

        Args:
            heartbeat_column: Column name to use for EOL pattern detection

        Returns:
            Self for method chaining
        """
        self.heartbeat_column = heartbeat_column
        return self

    def with_patterns(self, patterns: Dict[str, List[int]]) -> "SurvivalDatasetBuilder":
        """
        Set the patterns for EOL pattern detection.

        Args:
            patterns: Dict mapping pattern names to pattern definitions

        Returns:
            Self for method chaining
        """
        self.patterns = patterns
        return self

    def with_time_column(self, time_column: str) -> "SurvivalDatasetBuilder":
        """
        Set the time column for survival analysis.

        By default, if 'time_to_event' column exists in the data (generated by WindowAverager),
        it will be used regardless of this setting. Otherwise, this specified column will be used.

        Args:
            time_column: Column name to use as time-to-event

        Returns:
            Self for method chaining
        """
        self.time_column = time_column
        return self

    def with_sample_size(self, sample_size: int) -> "SurvivalDatasetBuilder":
        """
        Set the number of samples to use.

        Args:
            sample_size: Number of rows to process from the CSV

        Returns:
            Self for method chaining
        """
        self.sample_size = sample_size
        return self

    def with_debug(self, debug: bool = True) -> "SurvivalDatasetBuilder":
        """
        Set the debug flag.

        Args:
            debug: Whether to print additional debugging information

        Returns:
            Self for method chaining
        """
        self.debug = debug
        return self

    def with_window_size(self, window_size: int) -> "SurvivalDatasetBuilder":
        """
        Set the window size for calculating averages.

        Args:
            window_size: Number of values to include in each average window

        Returns:
            Self for method chaining
        """
        self.window_size = window_size
        return self

    def with_filter_short_events(
        self, filter_short_events: bool = True
    ) -> "SurvivalDatasetBuilder":
        """
        Set the filter_short_events flag.

        Args:
            filter_short_events: Whether to filter out records where time_to_event < window_size

        Returns:
            Self for method chaining
        """
        self.filter_short_events = filter_short_events
        return self

    def with_event_value(self, event_value: str) -> "SurvivalDatasetBuilder":
        """
        Set the event_value parameter which defines what value in the sample_type column
        represents the event (will be converted to event=1).

        Args:
            event_value: Value in sample_type column that represents the event

        Returns:
            Self for method chaining
        """
        self.event_value = event_value
        return self

    def with_focal_feature(
        self, focal_feature: Optional[str] = None
    ) -> "SurvivalDatasetBuilder":
        """
        Set the focal feature for interaction models. If provided, the builder will use
        the cox_model_with_interactions function to create the model. If None, it will use
        the basic_cox_model function.

        Args:
            focal_feature: The feature to interact with other features. Default is None.
                          Can use the original column name without the "_avg" suffix.

        Returns:
            Self for method chaining
        """
        # Store the original focal feature name
        self.focal_feature = focal_feature
        return self

    def get_available_columns(self) -> List[str]:
        """
        Get the list of available columns in the input file.

        This method allows users to see what columns are available before selecting
        target columns and heartbeat column.

        Returns:
            List of column names in the CSV file

        Raises:
            ValueError: If file path hasn't been set or the file doesn't exist
        """
        if self.file_path is None:
            raise ValueError(
                "File path must be set with with_file() before getting available columns"
            )

        if self.sample_builder is None:
            self._initialize_sample_builder()

        if self.sample_builder is None:
            raise ValueError("Failed to initialize SampleBuilder")

        if self._available_columns is None:
            self._available_columns = self.sample_builder.get_file_info()

        if self._available_columns is None:
            return []  # Return empty list if no columns are available

        return self._available_columns

    def add_pre_process_hook(
        self, hook: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> "SurvivalDatasetBuilder":
        """
        Add a pre-processing hook to transform the data before conversion.

        Args:
            hook: A function that takes a DataFrame and returns a transformed DataFrame

        Returns:
            Self for method chaining
        """
        self.pre_process_hooks.append(hook)
        return self

    def add_post_process_hook(
        self,
        hook: Callable[
            [pd.DataFrame, pd.Series, pd.Series],
            Tuple[pd.DataFrame, pd.Series, pd.Series],
        ],
    ) -> "SurvivalDatasetBuilder":
        """
        Add a post-processing hook to transform the data after conversion.

        Args:
            hook: A function that takes X, T, E and returns transformed X, T, E

        Returns:
            Self for method chaining
        """
        self.post_process_hooks.append(hook)
        return self

    def _validate_input(self) -> None:
        """
        Validate that all required inputs are provided.

        Raises:
            ValueError: If required inputs are missing
        """
        if self.file_path is None:
            raise ValueError(
                "File path must be provided using with_file() or during initialization"
            )

        if not self.feature_columns:
            raise ValueError(
                "Feature columns must be provided using with_feature_columns() or during initialization"
            )

        if not self.heartbeat_column:
            raise ValueError(
                "Heartbeat column must be provided using with_heartbeat_column() or during initialization"
            )

        if not self.patterns:
            raise ValueError(
                "Patterns dictionary must be provided using with_patterns() or during initialization"
            )

    def _initialize_sample_builder(self) -> None:
        """
        Initialize the SampleBuilder instance.
        """
        if self.file_path is None:
            return

        if self.debug:
            print(f"Debug: Initializing SampleBuilder with file: {self.file_path}")

        self.sample_builder = SampleBuilder(self.file_path)

    def _check_columns(self) -> bool:
        """
        Check if the requested columns exist in the input file.

        Returns:
            True if all columns exist, False otherwise
        """
        if self.sample_builder is None:
            self._initialize_sample_builder()

        if self.feature_columns is None:
            raise ValueError(
                "Feature columns must be specified before building the dataset"
            )

        available_columns = self.get_available_columns()
        if self.debug:
            print(f"Available columns in file: {available_columns}")
            print(f"Feature columns requested: {self.feature_columns}")

        # Check if target columns exist in file
        missing_columns = [
            col for col in self.feature_columns if col not in available_columns
        ]

        if missing_columns:
            warning_msg = (
                "The following columns were not found in the data: "
                + ", ".join(missing_columns)
            )
            if self.debug:
                print(f"WARNING: {warning_msg}")
            return False

        return True

    def _build_samples(self) -> None:
        """
        Build samples using the SampleBuilder.

        Raises:
            ValueError: If SampleBuilder fails to produce samples
        """
        if self.sample_builder is None:
            self._initialize_sample_builder()

        if self.sample_builder is None:
            raise ValueError("Failed to initialize SampleBuilder")

        if self.feature_columns is None:
            raise ValueError("Feature columns must be specified")

        if self.heartbeat_column is None:
            raise ValueError("Heartbeat column must be specified")

        if self.patterns is None:
            raise ValueError("Patterns must be specified")

        try:
            if self.debug:
                print("Building samples from SampleBuilder...")

            self.combined_df = self.sample_builder.build_and_combine_samples(
                target_columns=self.feature_columns,
                heartbeat_column=self.heartbeat_column,
                window_size=self.window_size,
                patterns=self.patterns,
                sample_size=self.sample_size,
            )

            if self.debug:
                if self.combined_df is not None and self.combined_df.empty:
                    print(
                        "Debug: SampleBuilder produced empty DataFrame. No samples matched the patterns."
                    )
                elif self.combined_df is not None:
                    print(
                        f"Debug: SampleBuilder produced {len(self.combined_df)} samples"
                    )
                    print(f"Sample columns: {self.combined_df.columns.tolist()}")

                    # Check if time_to_event column exists
                    if "time_to_event" in self.combined_df.columns:
                        print(
                            f"Debug: 'time_to_event' column found with values ranging from {self.combined_df['time_to_event'].min()} to {self.combined_df['time_to_event'].max()}"
                        )

                        # Automatically set time_column to time_to_event to ensure it's used
                        if self.time_column != "time_to_event":
                            print(
                                f"Debug: Setting time_column to 'time_to_event' (was {self.time_column})"
                            )
                            self.time_column = "time_to_event"
                    else:
                        print(
                            "Debug: 'time_to_event' column not found in combined data"
                        )
                else:
                    print("Debug: SampleBuilder produced None")

            # Update feature_columns to use the transformed column names with _avg suffix
            if self.combined_df is not None and not self.combined_df.empty:
                transformed_feature_cols = [
                    f"{col}_avg" for col in self.feature_columns
                ]
                # Verify these columns exist in the DataFrame
                existing_transformed_cols = [
                    col
                    for col in transformed_feature_cols
                    if col in self.combined_df.columns
                ]

                if existing_transformed_cols:
                    if self.debug:
                        print(
                            f"Debug: Updating feature columns to use transformed names with _avg suffix"
                        )
                        print(
                            f"Debug: Original feature columns: {self.feature_columns}"
                        )
                        print(
                            f"Debug: Updated feature columns: {existing_transformed_cols}"
                        )
                    self.feature_columns = existing_transformed_cols
                else:
                    print(
                        "Warning: No transformed feature columns (_avg suffix) found in the processed data"
                    )

        except Exception as e:
            if self.debug:
                print(f"Debug: Error in SampleBuilder: {e}")
            raise

    def _apply_pre_process_hooks(self) -> None:
        """
        Apply all pre-processing hooks to the combined DataFrame.
        """
        if not self.pre_process_hooks or self.combined_df is None:
            return

        for hook in self.pre_process_hooks:
            if self.debug:
                print(f"Applying pre-process hook: {hook.__name__}")
            self.combined_df = hook(self.combined_df)

    def _filter_metadata_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out metadata columns that should not be included in the feature matrix.

        Args:
            X: DataFrame containing feature columns

        Returns:
            DataFrame with metadata columns removed
        """
        columns_to_exclude = ["user_id", "eol_index"]

        # Check if columns exist before removing
        columns_to_drop = [col for col in columns_to_exclude if col in X.columns]

        if columns_to_drop:
            if self.debug:
                print(f"Removing metadata columns from features: {columns_to_drop}")
            return X.drop(columns=columns_to_drop)
        return X

    def _convert_to_lifelines_format(self) -> None:
        """
        Convert the combined DataFrame to lifelines format (X, T, E).
        """
        if self.combined_df is None:
            raise ValueError("No data available for conversion")

        if self.debug:
            print("Converting to lifelines format...")

        self.X, self.T, self.E = convert_to_lifelines_format(
            data=self.combined_df,
            time_column=self.time_column,
            event_column="sample_type",  # Fixed to use column name created by SampleBuilder
            event_value=self.event_value,
            feature_columns=self.feature_columns,
            time_from_window=self.time_from_window,
        )

        # Filter out metadata columns from features
        if self.X is not None:
            self.X = self._filter_metadata_columns(self.X)

        # Filter out records where time_to_event < window_size if enabled
        if (
            self.filter_short_events
            and self.T is not None
            and self.E is not None
            and self.X is not None
        ):
            original_samples = len(self.T)
            mask = self.T >= self.window_size
            self.X = self.X.loc[mask]
            self.T = self.T.loc[mask]
            self.E = self.E.loc[mask]

            if self.debug:
                filtered_samples = original_samples - len(self.T)
                if filtered_samples > 0:
                    print(
                        f"Filtered out {filtered_samples} samples with time_to_event < {self.window_size}"
                    )

        if (
            self.debug
            and self.X is not None
            and self.T is not None
            and self.E is not None
        ):
            print(
                f"Converted data: X shape {self.X.shape}, {self.E.sum()} events out of {len(self.E)} samples"
            )

    def _apply_post_process_hooks(self) -> None:
        """
        Apply all post-processing hooks to X, T, E.
        """
        if self.post_process_hooks:
            for hook in self.post_process_hooks:
                if self.debug:
                    print(f"Applying post-process hook: {hook.__name__}")
                self.X, self.T, self.E = hook(self.X, self.T, self.E)

    def build(self) -> CoxPHModel:
        """
        Build the survival dataset through the complete pipeline and return a fitted Cox model.

        Returns:
            CoxPHModel: Fitted Cox Proportional Hazards model.
            - If focal_feature is None, returns a basic Cox model
            - If focal_feature is specified, returns a Cox model with interactions

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Validate inputs
        self._validate_input()

        # Initialize SampleBuilder if not already initialized
        if self.sample_builder is None:
            self._initialize_sample_builder()

        # Check columns and warn if any are missing
        column_check_result = self._check_columns()
        if not column_check_result and not self.debug:
            print("WARNING: Some requested columns were not found in the data.")

        # Build samples using SampleBuilder
        self._build_samples()

        if self.combined_df is None or self.combined_df.empty:
            raise ValueError("Failed to build samples. No data available.")

        # Apply pre-processing hooks
        self._apply_pre_process_hooks()

        # Convert to lifelines format
        self._convert_to_lifelines_format()

        if self.X is None or self.T is None or self.E is None:
            raise ValueError("Failed to convert data to survival format")

        # Apply post-processing hooks
        self._apply_post_process_hooks()

        # Create and return the appropriate model based on focal_feature
        if self.focal_feature is not None:
            # Check if focal feature exists in the dataframe
            # First try the original focal feature name
            focal_feature_col = self.focal_feature

            # If the original name is not found, try adding "_avg" suffix
            if (
                focal_feature_col not in self.X.columns
                and f"{focal_feature_col}_avg" in self.X.columns
            ):
                focal_feature_col = f"{focal_feature_col}_avg"
                if self.debug:
                    print(f"Using transformed focal feature name: {focal_feature_col}")

            # Check if we found a valid focal feature column
            if focal_feature_col not in self.X.columns:
                raise ValueError(
                    f"Focal feature '{self.focal_feature}' or '{self.focal_feature}_avg' not found in dataframe columns: {list(self.X.columns)}"
                )

            model = cox_model_with_interactions(
                X=self.X, T=self.T, E=self.E, focal_feature=focal_feature_col
            )

            # Add additional information to the model
            model.heartbeat_column = self.heartbeat_column
            model.window_size = self.window_size
            if hasattr(self, "sample_size"):
                model.sample_size = self.sample_size

            return model
        else:
            model = basic_cox_model(X=self.X, T=self.T, E=self.E)

            # Add additional information to the model
            model.heartbeat_column = self.heartbeat_column
            model.window_size = self.window_size
            if hasattr(self, "sample_size"):
                model.sample_size = self.sample_size

            return model

    def get_feature_names(self) -> List[str]:
        """
        Get the names of the features used in the model.

        Returns:
            List of feature column names
        """
        if self.X is None:
            raise ValueError("Dataset has not been built yet. Call build() first.")
        # Explicitly convert to list to satisfy mypy
        return list(self.X.columns)

    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        if self.X is None or self.T is None or self.E is None:
            return {
                "status": "Not built",
                "message": "Dataset has not been built yet",
            }

        return {
            "status": "Built",
            "samples": len(self.X),
            "features": len(self.X.columns),
            "events": int(self.E.sum()),
            "censored": len(self.E) - int(self.E.sum()),
            "event_ratio": float(self.E.mean()),
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SurvivalDatasetBuilder":
        """
        Create a SurvivalDatasetBuilder from a configuration dictionary.

        Args:
            config: Dictionary containing configuration parameters

        Returns:
            SurvivalDatasetBuilder instance

        Raises:
            ValueError: If required configuration parameters are missing
        """
        # Validate required config parameters
        required_keys = ["feature_columns", "heartbeat_column", "patterns"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(
                f"Missing required configuration parameters: {', '.join(missing_keys)}"
            )

        builder = cls(
            feature_columns=config.get("feature_columns"),
            heartbeat_column=config.get("heartbeat_column"),
            patterns=config.get("patterns"),
            window_size=config.get("window_size", 5),
            event_value=config.get("event_value", "negative"),
            time_column=config.get("time_column"),
            time_from_window=config.get("time_from_window", True),
            debug=config.get("debug", False),
            filter_short_events=config.get("filter_short_events", True),
            focal_feature=config.get("focal_feature"),
        )

        # Set file path if provided
        if "file_path" in config:
            builder.with_file(config["file_path"])

        # Set sample size if provided
        if "sample_size" in config:
            builder.with_sample_size(config["sample_size"])

        # Set focal feature if provided
        if "focal_feature" in config:
            builder.with_focal_feature(config["focal_feature"])

        return builder
