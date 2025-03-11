from typing import Dict, List, Optional

import pandas as pd

# Change the imports to avoid circular imports
from ..loader.csv_processor import CSVProcessor
from ..samplers.eol_extractor import EOLExtractor
from ..samplers.window_averager import WindowAverager
from .log_utils import setup_logger

# Create logger for this module
logger = setup_logger("lifestay.loader.sample_builder")


class SampleBuilder:
    """
    SampleBuilder simplifies the process of creating positive and negative sample sets
    based on different EOL (End of Life) patterns.

    This class encapsulates the full data processing pipeline:
    1. CSV processing with CSVProcessor
    2. EOL pattern extraction with EOLExtractor
    3. Window average calculation with WindowAverager

    Samples are defined by different EOL patterns:
    - Negative samples: When default_pattern=[1, 0, 0] (or another specified pattern)
    - Positive samples: When alternative patterns are used
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialize the SampleBuilder with the input file path

        Args:
            file_path: Path to the CSV file to process
        """
        self.file_path = file_path
        self.csv_processor = CSVProcessor(file_path)
        self.window_averager = WindowAverager()
        self.processed_columns: List[str] = (
            []
        )  # Store column names of the processed DataFrame

        logger.info("SampleBuilder initialized")

    def get_file_info(self) -> List[str]:
        """
        Get information about the CSV file by reading only the header

        Returns:
            List of column names from the processed CSV
        """
        logger.info("Getting file information")

        # Use CSVProcessor to get only the column names
        column_names, original_names, _ = self.csv_processor.get_columns_info()

        # Store column names for future reference
        self.processed_columns = column_names

        logger.info(f"File has {len(column_names)} columns")
        return column_names

    def build_samples(
        self,
        target_columns: List[str],
        heartbeat_column: str,
        window_size: int,
        patterns: Dict[str, List[int]],
        sample_size: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Build sample sets (positive and negative) from a CSV file

        Args:
            target_columns: List of column names to calculate window averages for
            heartbeat_column: Column name to use for EOL pattern detection
            window_size: Number of values to include in each average window
            patterns: Dict mapping pattern names to pattern definitions
            sample_size: Optional number of rows to process (for testing)

        Returns:
            Dictionary mapping pattern names to resulting DataFrames with window averages
        """
        logger.info("Building sample sets")

        # Process the CSV file
        logger.info("Step 1: Processing CSV file")
        processed_df = self.csv_processor.process_csv(sample_size=sample_size)

        # Update our stored column names
        self.processed_columns = list(processed_df.columns)

        # Build samples for each pattern
        result_dfs = {}
        for pattern_name, pattern in patterns.items():
            logger.info(f"Building {pattern_name} samples with pattern {pattern}")

            # Extract EOL indices using the pattern
            eol_extractor = EOLExtractor(default_pattern=pattern)
            df_with_eol = eol_extractor.find_eol_patterns(
                processed_df, heartbeat_column
            )

            # Create a temporary copy to avoid data type warnings
            df_temp = df_with_eol.copy()

            # Ensure first_nonzero_idx is integer type
            first_nonzero_column = f"{heartbeat_column}_first_nonzero_idx"
            if first_nonzero_column in df_temp.columns:
                logger.info(
                    f"Found {first_nonzero_column} column, using it for time_to_event calculation"
                )
                # Convert first_nonzero_idx to int for non-null values
                non_null_mask = df_temp[first_nonzero_column].notnull()
                if non_null_mask.any():
                    df_temp.loc[non_null_mask, first_nonzero_column] = df_temp.loc[
                        non_null_mask, first_nonzero_column
                    ].astype(int)

            # Calculate window averages
            eol_indices_column = f"{heartbeat_column}_eol_indices"
            df_with_windows = self.window_averager.calculate_window_averages(
                df=df_temp,  # Use the temp dataframe with integer types
                target_columns=target_columns,
                eol_column=eol_indices_column,
                window_size=window_size,
                first_nonzero_column=(
                    first_nonzero_column
                    if first_nonzero_column in df_temp.columns
                    else None
                ),
            )

            # Add sample type column for identification
            df_with_windows["sample_type"] = pattern_name

            # Add to results dictionary
            result_dfs[pattern_name] = df_with_windows

            logger.info(f"Created {len(df_with_windows)} {pattern_name} samples")

        return result_dfs

    def combine_samples(self, sample_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple sample DataFrames into a single DataFrame

        Args:
            sample_dfs: Dictionary mapping pattern names to sample DataFrames

        Returns:
            Combined DataFrame with all samples
        """
        if not sample_dfs:
            return pd.DataFrame()

        logger.info(f"Combining {len(sample_dfs)} sample sets")

        # Concatenate all DataFrames
        combined_df = pd.concat(sample_dfs.values(), ignore_index=True)

        logger.info(f"Combined DataFrame has {len(combined_df)} rows")
        return combined_df

    def build_and_combine_samples(
        self,
        target_columns: List[str],
        heartbeat_column: str,
        window_size: int,
        patterns: Dict[str, List[int]],
        sample_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Build sample sets and combine them into a single DataFrame

        Args:
            target_columns: List of column names to calculate window averages for
            heartbeat_column: Column name to use for EOL pattern detection
            window_size: Number of values to include in each average window
            patterns: Dict mapping pattern names to pattern definitions
            sample_size: Optional number of rows to process (for testing)

        Returns:
            Combined DataFrame with all samples
        """
        # Build individual sample sets
        sample_dfs = self.build_samples(
            target_columns=target_columns,
            heartbeat_column=heartbeat_column,
            window_size=window_size,
            patterns=patterns,
            sample_size=sample_size,
        )

        # Combine them
        combined_df = self.combine_samples(sample_dfs)

        return combined_df
