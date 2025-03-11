"""
Data Converter for Survival Analysis

This module provides utilities to convert data from SampleBuilder format
to a format suitable for survival analysis with lifelines.
"""

from typing import List, Optional, Tuple

import pandas as pd


def convert_to_lifelines_format(
    data: pd.DataFrame,
    time_column: Optional[str] = None,
    event_column: str = "sample_type",  # Parameter retained for compatibility, but changes not recommended
    event_value: str = "negative",
    feature_columns: Optional[List[str]] = None,
    time_from_window: bool = True,
    window_size_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Convert data from SampleBuilder format to a format suitable for lifelines.

    In survival analysis:
    - 'duration' represents the time until the event occurs (or until censoring)
    - 'event' is a binary indicator (1 if the event occurred, 0 if censored)

    Args:
        data: DataFrame created by SampleBuilder
        time_column: Column to use as time-to-event. If None and time_from_window is True,
                    will generate based on window distance to event
        event_column: Column that indicates the event type (default is "sample_type", created by SampleBuilder,
                    it's recommended not to modify this default value, this parameter is kept for backward compatibility)
        event_value: Value in event_column that represents the event (will be converted to event=1)
        feature_columns: List of columns to use as features. If None, will use all numeric columns
                        excluding time and event columns
        time_from_window: Whether to generate time values based on window distance to the event
        window_size_column: Column that contains window size information if needed for time calculation

    Returns:
        Tuple of (X, T, E) where:
        - X: DataFrame containing feature columns
        - T: Series containing duration/time values
        - E: Series containing event indicators (1 = event occurred, 0 = censored)
    """
    if data.empty:
        raise ValueError(
            "Input data is empty. This likely means SampleBuilder didn't find any data matching your patterns or there's an issue with the input file or parameters."
        )

    # Create a copy to avoid modifying the original data
    df = data.copy()

    # Process event indicator
    # In survival analysis, event=1 means the event occurred (typically failure/death)
    # and event=0 means the observation was censored (event not observed)
    event = (df[event_column] == event_value).astype(int)

    # Process time/duration
    # First check if 'time_to_event' column exists, if it does, use it as priority
    if "time_to_event" in df.columns:
        # Ensure time_to_event is integer type
        duration = df["time_to_event"].astype(int)
    elif time_column is not None:
        # Use the specified time column
        if time_column not in df.columns:
            raise ValueError(f"Time column '{time_column}' not found in data")
        # If the specified time column is also 'time_to_event', ensure it's integer type
        if time_column == "time_to_event":
            duration = df[time_column].astype(int)
        else:
            duration = df[time_column]
    elif time_from_window:
        # Generate time based on window indices
        # This assumes data from SampleBuilder has window indices
        window_cols = [
            col for col in df.columns if "window_" in col and col.endswith("_idx")
        ]
        if not window_cols:
            raise ValueError(
                "No window index columns found in data and no time column specified"
            )

        # Use the first window column as reference
        # Typically, window_0_idx is closest to the event, window_1_idx is second closest, etc.
        # So we can use the index number as a proxy for time-to-event
        window_indices = pd.DataFrame({col: df[col] for col in window_cols})

        # Calculate time based on window index
        # The higher the window index, the further from the event
        # Extract the window number from column names like "window_0_idx" -> 0
        window_numbers = [int(col.split("_")[1]) for col in window_cols]
        max_window = max(window_numbers)

        # Time = max_window - window_number (so that time increases approaching the event)
        # Ensure the generated time values are integer type
        duration = pd.Series(max_window, index=df.index).astype(int)
    else:
        raise ValueError(
            "Either time_column must be specified or time_from_window must be True"
        )

    # Process feature columns
    if feature_columns is None:
        # Use all numeric columns excluding time and event columns
        exclude_cols = [event_column]
        if time_column is not None:
            exclude_cols.append(time_column)

        # Ensure 'time_to_event' column is also excluded
        if "time_to_event" in df.columns:
            exclude_cols.append("time_to_event")

        # Exclude window index columns
        exclude_cols.extend([col for col in df.columns if "_idx" in col])

        # Exclude metadata columns that should not be used as features
        metadata_cols = ["user_id", "eol_index"]
        exclude_cols.extend([col for col in metadata_cols if col in df.columns])

        # Get remaining numeric columns
        feature_columns = [
            col
            for col in df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
    else:
        # Even when feature_columns is specified, ensure user_id and eol_index are excluded
        metadata_cols = ["user_id", "eol_index"]
        feature_columns = [col for col in feature_columns if col not in metadata_cols]

    # Validate feature columns
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Feature columns {missing_cols} not found in data")

    # Create feature matrix
    X = df[feature_columns]

    return X, duration, event
