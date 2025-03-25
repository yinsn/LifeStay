"""
Time-to-First Event Dataset Builder Module

This module provides a class-based approach to building time-to-first-event analysis datasets
from time series data, using TimeToFirstEventBuilder for event detection.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..loader.csv_processor import CSVProcessor
from ..loader.sample_builder import SampleBuilder
from ..samplers.time_to_first_event import TimeToFirstEventBuilder


class TimeToFirstEventPipeline:
    """
    Pipeline class for creating time-to-first-event analysis datasets from CSV files.

    This class handles the pipeline for preparing event analysis data:
    1. Loading and processing CSV data
    2. Detecting the first occurrence of decline events using TimeToFirstEventBuilder
    3. Preparing data for survival analysis (X, T, E format)

    Usage example:
        pipeline = TimeToFirstEventPipeline() \\
            .with_file("data.csv") \\
            .with_feature_columns(["col1", "col2"]) \\
            .with_heartbeat_column("heartbeat") \\
            .with_window_size(5) \\
            .with_drop_threshold(0.2) \\
            .with_debug(True)

        pipeline.build()

    Attributes:
        file_path: Path to the input CSV file
        feature_columns: List of column names to use as features
        heartbeat_column: Column name for heartbeat/activity data to detect events
        window_size: Number of values to include in each average window
        drop_threshold: Threshold percentage for decline (default 0.2 = 20%)
        min_avg_value: Minimum average value to consider for event triggering
        max_avg_value: Maximum average value to consider for event triggering
        debug: Whether to print debug information
    """

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        heartbeat_column: Optional[str] = None,
        window_size: int = 5,
        drop_threshold: float = 0.2,
        min_avg_value: float = 0.0,
        max_avg_value: float = float("inf"),
        debug: bool = False,
    ):
        """
        Initialize the TimeToFirstEventPipeline.

        Args:
            feature_columns: List of column names to use as features
            heartbeat_column: Column name to use for event detection
            window_size: Number of values to include in each average window
            drop_threshold: Threshold percentage for value decline (0.2 = 20%)
            min_avg_value: Minimum average value to consider for event triggering
            max_avg_value: Maximum average value to consider for event triggering
            debug: Whether to print debug information
        """
        self.file_path: Optional[str] = None
        self.feature_columns: Optional[List[str]] = feature_columns
        self.heartbeat_column: Optional[str] = heartbeat_column
        self.window_size: int = window_size
        self.drop_threshold: float = drop_threshold
        self.min_avg_value: float = min_avg_value
        self.max_avg_value: float = max_avg_value
        self.debug: bool = debug

        # Data processing instances
        self.csv_processor: Optional[CSVProcessor] = None
        self.sample_builder: Optional[SampleBuilder] = None

        # Cache for available columns
        self._available_columns: Optional[List[str]] = None

        # Data containers
        self.processed_df: Optional[pd.DataFrame] = None
        self.individual_data: Optional[Dict[str, List[float]]] = None
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

    def __call__(self, file_path: str, sample_size: Optional[int] = None) -> None:
        """
        Build the event dataset when the instance is called as a function.

        Args:
            file_path: Path to the CSV file to process
            sample_size: Optional number of rows to process
        """
        if self.file_path is None or self.file_path != file_path:
            self.file_path = file_path

        self.sample_size = sample_size
        self.build()

    def with_file(self, file_path: str) -> "TimeToFirstEventPipeline":
        """
        Set the input file path for the dataset.

        Args:
            file_path: Path to the CSV file to process

        Returns:
            Self for method chaining
        """
        self.file_path = file_path
        self._initialize_processors()
        return self

    def with_feature_columns(
        self, feature_columns: List[str]
    ) -> "TimeToFirstEventPipeline":
        """
        Set the feature columns to use.

        Args:
            feature_columns: List of column names to use as features in the model

        Returns:
            Self for method chaining
        """
        self.feature_columns = feature_columns
        return self

    def with_heartbeat_column(
        self, heartbeat_column: str
    ) -> "TimeToFirstEventPipeline":
        """
        Set the heartbeat column for event detection.

        Args:
            heartbeat_column: Column name to use for event detection

        Returns:
            Self for method chaining
        """
        self.heartbeat_column = heartbeat_column
        return self

    def with_window_size(self, window_size: int) -> "TimeToFirstEventPipeline":
        """
        Set the window size for calculating averages.

        Args:
            window_size: Number of values to include in each average window

        Returns:
            Self for method chaining
        """
        self.window_size = window_size
        return self

    def with_drop_threshold(self, drop_threshold: float) -> "TimeToFirstEventPipeline":
        """
        Set the drop threshold for event detection.

        Args:
            drop_threshold: Threshold percentage for value decline (0.2 = 20%)

        Returns:
            Self for method chaining
        """
        self.drop_threshold = drop_threshold
        return self

    def with_min_avg_value(self, min_avg_value: float) -> "TimeToFirstEventPipeline":
        """
        Set the minimum average value to consider for event triggering.

        Args:
            min_avg_value: Minimum average value to consider for event triggering

        Returns:
            Self for method chaining
        """
        self.min_avg_value = min_avg_value
        return self

    def with_max_avg_value(self, max_avg_value: float) -> "TimeToFirstEventPipeline":
        """
        Set the maximum average value to consider for event triggering.

        Args:
            max_avg_value: Maximum average value to consider for event triggering

        Returns:
            Self for method chaining
        """
        self.max_avg_value = max_avg_value
        return self

    def with_debug(self, debug: bool = True) -> "TimeToFirstEventPipeline":
        """
        Set the debug flag.

        Args:
            debug: Whether to print additional debugging information

        Returns:
            Self for method chaining
        """
        self.debug = debug
        return self

    def with_sample_size(self, sample_size: int) -> "TimeToFirstEventPipeline":
        """
        Set the number of samples to use.

        Args:
            sample_size: Number of rows to process from the CSV

        Returns:
            Self for method chaining
        """
        self.sample_size = sample_size
        return self

    def add_pre_process_hook(
        self, hook: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> "TimeToFirstEventPipeline":
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
    ) -> "TimeToFirstEventPipeline":
        """
        Add a post-processing hook to transform the data after conversion.

        Args:
            hook: A function that takes X, T, E and returns transformed X, T, E

        Returns:
            Self for method chaining
        """
        self.post_process_hooks.append(hook)
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
            self._initialize_processors()

        if self.sample_builder is None:
            raise ValueError("Failed to initialize SampleBuilder")

        if self._available_columns is None:
            self._available_columns = self.sample_builder.get_file_info()

        if self._available_columns is None:
            return []  # Return empty list if no columns are available

        return self._available_columns

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

    def _initialize_processors(self) -> None:
        """
        Initialize the CSV processor and SampleBuilder instance.
        """
        if self.file_path is None:
            return

        if self.debug:
            print(f"Debug: Initializing processors with file: {self.file_path}")

        # Initialize SampleBuilder for column info
        self.sample_builder = SampleBuilder(self.file_path)

        # Initialize CSVProcessor
        self.csv_processor = CSVProcessor(self.file_path)

    def _check_columns(self) -> bool:
        """
        Check if the requested columns exist in the input file.

        Returns:
            True if all columns exist, False otherwise
        """
        if self.sample_builder is None:
            self._initialize_processors()

        if self.feature_columns is None:
            raise ValueError(
                "Feature columns must be specified before building the dataset"
            )

        available_columns = self.get_available_columns()
        if self.debug:
            print(f"Available columns in file: {available_columns}")
            print(f"Feature columns requested: {self.feature_columns}")
            print(f"Heartbeat column: {self.heartbeat_column}")

        # Check if target columns exist in file
        missing_columns = [
            col for col in self.feature_columns if col not in available_columns
        ]

        # Check if heartbeat column exists
        if self.heartbeat_column and self.heartbeat_column not in available_columns:
            missing_columns.append(self.heartbeat_column)

        if missing_columns:
            warning_msg = (
                "The following columns were not found in the data: "
                + ", ".join(missing_columns)
            )
            if self.debug:
                print(f"WARNING: {warning_msg}")
            return False

        return True

    def _preprocess_arrays(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess arrays in DataFrame, convert feature columns to scalar values.
        Keeps the heartbeat column as arrays for proper handling in _extract_individual_data.

        Args:
            df: Input DataFrame with array columns

        Returns:
            Processed DataFrame with feature arrays converted to scalar values
        """
        if self.debug:
            print("Preprocessing arrays in DataFrame...")

        # Create a copy of the DataFrame to avoid modifying the original
        processed_df = df.copy()

        # Inspect the first row to check for array data
        if self.heartbeat_column and self.heartbeat_column in processed_df.columns:
            first_element = processed_df[self.heartbeat_column].iloc[0]
            if self.debug and isinstance(first_element, (np.ndarray, list)):
                print(
                    f"First element of {self.heartbeat_column}: {first_element}, type: {type(first_element)}"
                )

        for column in processed_df.columns:
            if (
                column == "user_id" or len(processed_df) == 0
            ):  # Skip ID column or empty DataFrame
                continue

            # Check if this column contains arrays/lists
            try:
                first_val = processed_df[column].iloc[0]
                if isinstance(first_val, (np.ndarray, list)):
                    if self.debug:
                        print(
                            f"Converting array column: {column}, array shape: {np.shape(first_val)}"
                        )

                    # Keep the heartbeat column as a numpy array
                    if self.heartbeat_column and column == self.heartbeat_column:
                        continue
                    else:
                        # For feature columns, take the sum of values in each array
                        processed_df[column] = processed_df[column].apply(
                            lambda x: (
                                float(np.sum(x))
                                if isinstance(x, (np.ndarray, list)) and len(x) > 0
                                else 0.0
                            )
                        )
            except Exception as e:
                if self.debug:
                    print(f"Error processing column {column}: {e}")

        return processed_df

    def _process_data(self) -> None:
        """
        Process the CSV data and prepare it for event detection.
        """
        if self.csv_processor is None:
            self._initialize_processors()

        if self.csv_processor is None:
            raise ValueError("Failed to initialize CSVProcessor")

        try:
            if self.debug:
                print("Processing CSV data...")

            # Process the CSV file
            self.processed_df = self.csv_processor.process_csv(
                sample_size=self.sample_size
            )

            if self.processed_df is None or self.processed_df.empty:
                raise ValueError("CSV processing returned empty DataFrame")

            if self.debug:
                print(f"Processed {len(self.processed_df)} rows from CSV")

            # Apply array preprocessing first
            self.processed_df = self._preprocess_arrays(self.processed_df)

            # Apply pre-processing hooks
            self._apply_pre_process_hooks()

            # Extract relevant data for each individual
            self._extract_individual_data()

        except Exception as e:
            if self.debug:
                print(f"Error in data processing: {e}")
            raise

    def _extract_individual_data(self) -> None:
        """
        Extract individual time series data for event detection.
        Handles both regular values and numpy arrays/lists.
        """
        if self.processed_df is None or self.heartbeat_column is None:
            return

        if self.debug:
            print("Extracting individual time series data...")

        # Create a dictionary mapping individual IDs to their time series data
        individual_data = {}

        # Assume the first column is the user/individual ID
        # This might need to be parameterized in the future
        id_column = self.processed_df.columns[0]

        for individual_id, group in self.processed_df.groupby(id_column):
            # Get the heartbeat data for this individual
            heartbeat_data = group[self.heartbeat_column].iloc[0]

            # Ensure heartbeat_data is a list of floats
            if isinstance(heartbeat_data, np.ndarray):
                heartbeat_data = heartbeat_data.tolist()
            elif not isinstance(heartbeat_data, list):
                # For regular (non-list) data, convert to list
                if isinstance(heartbeat_data, (list, np.ndarray)):
                    # If it's already an iterable, convert to list
                    heartbeat_data = list(heartbeat_data)
                else:
                    # If it's a scalar, make it a single-item list
                    heartbeat_data = [heartbeat_data]

            # Ensure all values are floats and handle NaN values
            heartbeat_data = [float(x) if pd.notna(x) else 0.0 for x in heartbeat_data]

            individual_data[str(individual_id)] = heartbeat_data

        self.individual_data = individual_data

        if self.debug:
            print(f"Extracted time series data for {len(individual_data)} individuals")
            # Print the first few values of the first individual if available
            if individual_data:
                first_id = next(iter(individual_data))
                first_values = (
                    individual_data[first_id][:5]
                    if len(individual_data[first_id]) >= 5
                    else individual_data[first_id]
                )
                print(f"First individual ID: {first_id}, first values: {first_values}")

    def _detect_first_events(self) -> None:
        """
        Detect first occurrence of events for each individual using TimeToFirstEventBuilder.
        """
        if self.individual_data is None:
            raise ValueError("No individual data available for event detection")

        if self.debug:
            print("Detecting first events using TimeToFirstEventBuilder...")

        # Create the TimeToFirstEventBuilder
        event_builder = TimeToFirstEventBuilder(
            window_size=self.window_size,
            drop_threshold=self.drop_threshold,
            min_avg_value=self.min_avg_value,
            max_avg_value=self.max_avg_value,
        )

        # Generate first event data
        first_event_data = event_builder.generate_first_event_data(self.individual_data)

        if self.debug:
            print(f"Generated first event data for {len(first_event_data)} individuals")

        # Create a DataFrame from the first event data
        df_events = pd.DataFrame(first_event_data)

        # Extract features for each individual
        if self.feature_columns and self.processed_df is not None:
            # Join features with event data
            # First, calculate average feature values for each individual
            id_column = self.processed_df.columns[0]
            feature_avgs = self.processed_df.groupby(id_column)[
                self.feature_columns
            ].mean()

            # Convert feature_avgs index to string to match individual_id in df_events
            feature_avgs.index = feature_avgs.index.astype(str)

            # Join with df_events
            df_events = (
                df_events.set_index("individual_id").join(feature_avgs).reset_index()
            )

        # Prepare X, T, E
        self.X = (
            df_events[self.feature_columns] if self.feature_columns else pd.DataFrame()
        )
        self.T = df_events["observation_time"]
        self.E = df_events["event_indicator"]

        if self.debug:
            print(
                f"Prepared data: X shape {self.X.shape}, {self.E.sum()} events out of {len(self.E)} samples"
            )

    def _apply_pre_process_hooks(self) -> None:
        """
        Apply all pre-processing hooks to the processed DataFrame.
        """
        if not self.pre_process_hooks or self.processed_df is None:
            return

        for hook in self.pre_process_hooks:
            if self.debug:
                print(f"Applying pre-process hook: {hook.__name__}")
            self.processed_df = hook(self.processed_df)

    def _apply_post_process_hooks(self) -> None:
        """
        Apply all post-processing hooks to X, T, E.
        """
        if (
            not self.post_process_hooks
            or self.X is None
            or self.T is None
            or self.E is None
        ):
            return

        for hook in self.post_process_hooks:
            if self.debug:
                print(f"Applying post-process hook: {hook.__name__}")
            self.X, self.T, self.E = hook(self.X, self.T, self.E)

    def build(self) -> None:
        """
        Build the time-to-first-event dataset.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Validate inputs
        self._validate_input()

        # Check columns and warn if any are missing
        column_check_result = self._check_columns()
        if not column_check_result and not self.debug:
            print("WARNING: Some requested columns were not found in the data.")

        # Process the data
        self._process_data()

        # Detect first events
        self._detect_first_events()

        if self.X is None or self.T is None or self.E is None:
            raise ValueError("Failed to create survival data")

        # Apply post-processing hooks
        self._apply_post_process_hooks()

        # No model creation or return needed
        if self.debug:
            print("Dataset built successfully")

    def get_feature_names(self) -> List[str]:
        """
        Get the names of the features used in the dataset.

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
    def from_config(cls, config: Dict[str, Any]) -> "TimeToFirstEventPipeline":
        """
        Create a TimeToFirstEventPipeline from a configuration dictionary.

        Args:
            config: Dictionary containing configuration parameters

        Returns:
            TimeToFirstEventPipeline instance

        Raises:
            ValueError: If required configuration parameters are missing
        """
        # Validate required config parameters
        required_keys = ["feature_columns", "heartbeat_column"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(
                f"Missing required configuration parameters: {', '.join(missing_keys)}"
            )

        builder = cls(
            feature_columns=config.get("feature_columns"),
            heartbeat_column=config.get("heartbeat_column"),
            window_size=config.get("window_size", 5),
            drop_threshold=config.get("drop_threshold", 0.2),
            min_avg_value=config.get("min_avg_value", 0.0),
            max_avg_value=config.get("max_avg_value", float("inf")),
            debug=config.get("debug", False),
        )

        # Set file path if provided
        if "file_path" in config:
            builder.with_file(config["file_path"])

        # Set sample size if provided
        if "sample_size" in config:
            builder.with_sample_size(config["sample_size"])

        return builder
