from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .log_utils import setup_logger

# Create logger for this module
logger = setup_logger("lifestay.samplers.window_averager")


class WindowAverager:
    """
    WindowAverager calculates the average values of specified columns
    for windows preceding EOL (End of Life) indices.

    This is useful for analyzing patterns that occur before activity stops.
    """

    def __init__(self) -> None:
        """Initialize the window averager"""

    def calculate_window_averages(
        self,
        df: pd.DataFrame,
        target_columns: List[str],
        eol_column: str,
        window_size: int,
        first_nonzero_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Calculate window averages for specified columns based on EOL indices

        Args:
            df: DataFrame containing numpy arrays and EOL indices
            target_columns: List of column names to calculate averages for
            eol_column: Column name containing EOL indices
            window_size: Number of values to include in each average window
            first_nonzero_column: Column name containing first non-zero indices (optional)

        Returns:
            DataFrame with window averages for each EOL index and target column,
            and time to event if first_nonzero_column is provided
        """
        logger.info(f"Calculating window averages for {len(target_columns)} columns")
        logger.info(f"Window size: {window_size}")

        # Create an empty list to store results
        result_rows = []

        # Process each row in the DataFrame
        for i, row in enumerate(df.itertuples()):
            # Display progress for large datasets
            if i % 10000 == 0 and i > 0:
                logger.info(f"Processed {i} rows...")

            try:
                # Get the EOL indices for this row
                eol_indices = getattr(row, eol_column)

                # Check if there are any EOL indices
                if (
                    not isinstance(eol_indices, (list, np.ndarray))
                    or len(eol_indices) == 0
                ):
                    continue

                # Get the ID column (first column in the DataFrame)
                id_column = df.columns[0]
                row_id = getattr(row, id_column)

                # Get the first non-zero index if the column is provided
                first_nonzero_idx = None
                if first_nonzero_column is not None:
                    first_nonzero_idx = getattr(row, first_nonzero_column)

                # Process each EOL index
                for eol_idx in eol_indices:
                    # Skip if index is 0 (no preceding values)
                    if eol_idx == 0:
                        continue

                    # Create a row for this EOL index
                    eol_idx_int = int(eol_idx)
                    result_row = {id_column: row_id, "eol_index": eol_idx_int}

                    # Calculate time to event if first non-zero index is available
                    if first_nonzero_idx is not None and first_nonzero_idx is not None:
                        result_row["time_to_event"] = int(
                            eol_idx_int - first_nonzero_idx
                        )

                    # Calculate averages for each target column
                    for col_name in target_columns:
                        col_array = getattr(row, col_name)

                        # Skip if the column is not an array
                        if not isinstance(col_array, (list, np.ndarray)):
                            result_row[f"{col_name}_avg"] = None
                            continue

                        # Calculate the window average
                        window_avg = self._calculate_window_average(
                            col_array, eol_idx_int, window_size
                        )

                        # Add to result row
                        result_row[f"{col_name}_avg"] = window_avg

                    # Add the row to results
                    result_rows.append(result_row)

            except Exception as e:
                logger.error(f"Error processing row {i}: {e}")
                continue

        # Create a DataFrame from the results
        result_df = pd.DataFrame(result_rows)
        logger.info(
            f"Window average calculation complete. Created {len(result_df)} rows."
        )

        return result_df

    def _calculate_window_average(
        self, array: Union[List, np.ndarray], end_idx: int, window_size: int
    ) -> Optional[float]:
        """
        Calculate the average of values in the window preceding the end index

        Args:
            array: Array of values
            end_idx: End index (exclusive)
            window_size: Size of the window

        Returns:
            Average value in the window (rounded to 5 significant digits), or None if no values to average
        """
        # Calculate start index (ensuring it's not negative)
        start_idx = max(0, end_idx - window_size)

        # If there are no values to average, return None
        if start_idx == end_idx:
            return None

        # Extract the window values
        window_values = array[start_idx:end_idx]

        # Calculate average (ignoring NaN values) and round to 5 significant digits
        if isinstance(window_values, np.ndarray):
            # Handle numpy arrays
            avg = np.nanmean(window_values)
            return float(f"{avg:.5g}")
        else:
            # Handle Python lists
            values = [v for v in window_values if v is not None]
            if not values:
                return None
            avg = sum(values) / len(values)
            return float(f"{avg:.5g}")
