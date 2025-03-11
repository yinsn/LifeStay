from typing import List, Optional

import numpy as np
import pandas as pd

from .log_utils import setup_logger

# Create logger for this module
logger = setup_logger("lifestay.samplers.eol_extractor")


class EOLExtractor:
    """
    End of Life (EOL) extractor that identifies patterns in time-series data.

    By default, it finds patterns where a non-zero value is followed by two zeros (X, 0, 0)
    in a specified column of a DataFrame, but it can be configured to detect custom patterns
    like (X, X, 0), (X, X, X, 0), or (X, X, X).

    This pattern detection can indicate when activity stops or "dies off" in time-series data.
    """

    def __init__(self, default_pattern: Optional[List[int]] = None) -> None:
        """
        Initialize the EOL extractor with an optional default pattern

        Args:
            default_pattern: A list where 1 represents non-zero (X) and 0 represents zero.
                             Default is [1, 0, 0] for the pattern (X, 0, 0)
        """
        # Default pattern is [1, 0, 0] representing (X, 0, 0)
        self.default_pattern = (
            default_pattern if default_pattern is not None else [1, 0, 0]
        )

    def extract_eol_indices(
        self,
        df: pd.DataFrame,
        heartbeat_column: str,
        pattern: Optional[List[int]] = None,
    ) -> List[List[int]]:
        """
        Extract indices where the specified pattern occurs in the heartbeat column for each row

        Args:
            df: DataFrame containing numpy arrays in the heartbeat column
            heartbeat_column: Column name to analyze for patterns
            pattern: A list where 1 represents non-zero (X) and 0 represents zero.
                    If None, uses the default pattern from initialization.

        Returns:
            List of lists containing indices where the pattern occurs for each row
        """
        pattern_to_use = pattern if pattern is not None else self.default_pattern
        pattern_str = self._pattern_to_string(pattern_to_use)

        logger.info(
            f"Extracting indices for pattern {pattern_str} from column: {heartbeat_column}"
        )

        results = []

        # Process each row
        for i, row_array in enumerate(df[heartbeat_column]):
            # Display progress for large datasets
            if i % 10000 == 0 and i > 0:
                logger.info(f"Processed {i} rows...")

            # Extract patterns for this row
            try:
                # Check if it's a numpy array
                if isinstance(row_array, np.ndarray):
                    eol_indices = self._find_pattern(row_array, pattern_to_use)
                    results.append(eol_indices)
                else:
                    # If not an array, return empty list
                    logger.warning(
                        f"Row {i} doesn't contain a numpy array in column {heartbeat_column}"
                    )
                    results.append([])

            except Exception as e:
                logger.error(f"Error processing row {i}: {e}")
                results.append([])

        logger.info(
            f"Pattern extraction complete. Found patterns in {sum(1 for x in results if x)} out of {len(results)} rows."
        )
        return results

    def _pattern_to_string(self, pattern: List[int]) -> str:
        """Convert a pattern array to a readable string format"""
        return "(" + ", ".join("X" if p == 1 else "0" for p in pattern) + ")"

    def _find_pattern(self, arr: np.ndarray, pattern: List[int]) -> List[int]:
        """
        Find all indices where the specified pattern occurs

        Args:
            arr: Numpy array to analyze
            pattern: A list where 1 represents non-zero (X) and 0 represents zero

        Returns:
            List of indices where the pattern starts
        """
        # Return empty list for arrays that are too small
        pattern_length = len(pattern)
        if len(arr) < pattern_length:
            return []

        indices = []

        # Check each position for the pattern
        for i in range(len(arr) - pattern_length + 1):
            match = True
            for j, p in enumerate(pattern):
                # p=1 means non-zero (X), p=0 means zero
                if (p == 1 and arr[i + j] == 0) or (p == 0 and arr[i + j] != 0):
                    match = False
                    break

            if match:
                indices.append(i)

        return indices

    def _find_first_nonzero_index(self, arr: np.ndarray) -> Optional[int]:
        """
        Find the index of the first non-zero element in an array

        Args:
            arr: Numpy array to analyze

        Returns:
            Index of the first non-zero element as Python int, or None if all elements are zero
        """
        if not isinstance(arr, np.ndarray) or len(arr) == 0:
            return None

        # Find indices of non-zero elements
        nonzero_indices = np.nonzero(arr)[0]

        # Return the first index as Python int, or None if there are no non-zero elements
        if len(nonzero_indices) > 0:
            # Explicitly convert numpy.int64 to Python int
            return int(nonzero_indices[0])
        return None

    def find_eol_patterns(
        self,
        df: pd.DataFrame,
        heartbeat_column: str,
        pattern: Optional[List[int]] = None,
        result_column_suffix: str = "_eol_indices",
    ) -> pd.DataFrame:
        """
        Analyze the DataFrame and add a new column with pattern match indices

        Args:
            df: DataFrame containing numpy arrays in the heartbeat column
            heartbeat_column: Column name to analyze for patterns
            pattern: A list where 1 represents non-zero (X) and 0 represents zero.
                    If None, uses the default pattern from initialization.
            result_column_suffix: Suffix for the new column name (default: "_eol_indices")

        Returns:
            DataFrame with additional columns containing pattern indices and first non-zero indices
        """
        pattern_to_use = pattern if pattern is not None else self.default_pattern
        pattern_str = self._pattern_to_string(pattern_to_use)

        logger.info(f"Finding {pattern_str} patterns in column: {heartbeat_column}")

        # Create a copy of the DataFrame
        result_df = df.copy()

        # Extract pattern indices
        pattern_indices = self.extract_eol_indices(df, heartbeat_column, pattern_to_use)

        # Add as a new column
        result_df[f"{heartbeat_column}{result_column_suffix}"] = pattern_indices

        # Extract first non-zero indices
        first_nonzero_indices = []
        for i, row_array in enumerate(df[heartbeat_column]):
            try:
                if isinstance(row_array, np.ndarray):
                    first_idx = self._find_first_nonzero_index(row_array)
                    # _find_first_nonzero_index already returns Python integer type
                    first_nonzero_indices.append(first_idx)
                else:
                    first_nonzero_indices.append(None)
            except Exception as e:
                logger.error(f"Error finding first non-zero index for row {i}: {e}")
                first_nonzero_indices.append(None)

        # Create a Pandas Series, explicitly setting the type to Int64 (Pandas' nullable integer type)
        first_nonzero_col = f"{heartbeat_column}_first_nonzero_idx"
        # Convert to pandas Int64 type, which is an integer type that can contain NA values
        first_nonzero_series = pd.Series(
            first_nonzero_indices, index=df.index, dtype="Int64"
        )

        # Add as a new column
        result_df[first_nonzero_col] = first_nonzero_series

        logger.info(f"Pattern {pattern_str} extraction complete")
        return result_df
