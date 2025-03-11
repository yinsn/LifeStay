import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .log_utils import setup_logger

# Create logger for this module
logger = setup_logger("lifestay.loader.csv_processor")


class CSVProcessor:
    """
    Process CSV files containing lists by reading them and converting to numpy arrays
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialize the processor with file path and default regex pattern for lists

        Args:
            file_path: Path to the CSV file to process
        """
        # Store the file path
        self.file_path = file_path

        # Pattern matching for list fields
        self.list_pattern = re.compile(r"\[(.*?)\]")
        self.column_names: List[str] = []  # Processed column names
        self.original_column_names: List[str] = (
            []
        )  # Original column names before processing
        self.header_mapping: Dict[str, str] = (
            {}
        )  # Mapping from original to processed column names

        # Log file information
        if os.path.exists(file_path):
            logger.info(f"CSVProcessor initialized with file: {file_path}")
            logger.info(
                f"File size: {os.path.getsize(file_path) / (1024 * 1024 * 1024):.2f} GB"
            )
        else:
            logger.warning(f"File not found: {file_path}")

    def get_columns_info(self) -> Tuple[List[str], List[str], Dict[str, str]]:
        """
        Get column information from CSV file by reading only the header row

        Returns:
            Tuple containing:
                - List of processed column names
                - List of original column names
                - Dictionary mapping original column names to processed column names
        """
        logger.info("Reading column information")

        # Open file and read only the header
        with open(self.file_path, "r", encoding="utf-8") as f:
            # Read and process header row
            original_header = f.readline().strip().split(",")

            # Extract the part after the dot as the new header
            header = [
                col.split(".")[-1] if "." in col else col for col in original_header
            ]

            # Save the mapping between original and new headers
            header_mapping = dict(zip(original_header, header))

            # Store in instance variables
            self.column_names = header
            self.original_column_names = original_header
            self.header_mapping = header_mapping

            logger.info(f"Found {len(header)} columns in CSV file")

            return header, original_header, header_mapping

    def process_csv(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Process a CSV file containing lists by reading it and converting lists to numpy arrays

        Args:
            sample_size: Optional number of rows to process (for testing)

        Returns:
            DataFrame with lists converted to numpy arrays
        """
        logger.info("Starting to process file")

        # First read the CSV
        df = self._read_csv_with_lists(sample_size)

        # Identify the ID column (first column)
        id_column = df.columns[0]
        logger.info(f"Using '{id_column}' as the ID column")

        # Convert lists to numpy arrays
        result_df = self._convert_to_numpy(df, id_column)

        logger.info(
            f"Processing complete! DataFrame has {len(result_df)} rows, {len(result_df.columns)} columns"
        )
        return result_df

    def _read_csv_with_lists(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Read CSV file containing lists line by line

        Args:
            sample_size: If specified, only read n rows (for testing)

        Returns:
            pandas DataFrame
        """
        # List to store all data
        data: List[Dict[str, Any]] = []

        # Open file and read
        with open(self.file_path, "r", encoding="utf-8") as f:
            # Read and process header row
            original_header = f.readline().strip().split(",")

            # Extract the part after the dot as the new header
            header = [
                col.split(".")[-1] if "." in col else col for col in original_header
            ]

            id_col = header[0]  # First column is ID
            list_cols = header[1:]  # Other columns are lists

            # Save the mapping between original and new headers
            header_mapping = dict(zip(original_header, header))

            # Store in instance variables
            self.column_names = header
            self.original_column_names = original_header
            self.header_mapping = header_mapping

            # Read line by line
            for i, line in enumerate(f):
                # If only sample data is needed
                if sample_size and i >= sample_size:
                    break

                # Display progress
                if i % 10000 == 0 and i > 0:
                    logger.info(f"Processed {i} rows...")

                try:
                    # Create a new row data
                    row_data: Dict[str, Any] = {}

                    # Split the line, keeping commas inside brackets
                    parts = []
                    current_part = ""
                    in_brackets = False

                    for char in line.strip():
                        if char == "[":
                            in_brackets = True
                            current_part += char
                        elif char == "]":
                            in_brackets = False
                            current_part += char
                        elif char == "," and not in_brackets:
                            parts.append(current_part)
                            current_part = ""
                        else:
                            current_part += char

                    # Add the last part
                    if current_part:
                        parts.append(current_part)

                    # Ensure number of parts matches the header
                    if len(parts) != len(header):
                        logger.warning(
                            f"Warning: Row {i+2} has {len(parts)} columns, expected {len(header)}"
                        )
                        # If not enough columns, add null values
                        while len(parts) < len(header):
                            parts.append("null")
                        # If too many columns, truncate
                        if len(parts) > len(header):
                            parts = parts[: len(header)]

                    # Set ID column
                    row_data[id_col] = (
                        int(parts[0]) if parts[0] and parts[0] != "null" else None
                    )

                    # Process list columns
                    for j, col_name in enumerate(list_cols):
                        if j + 1 >= len(parts):  # Safety check
                            row_data[col_name] = None
                            continue

                        value = parts[j + 1]

                        # Handle empty values
                        if not value or value == "null":
                            row_data[col_name] = None
                            continue

                        # Handle lists
                        if value.startswith("[") and value.endswith("]"):
                            # Extract list content
                            list_content = self.list_pattern.search(value)
                            if list_content:
                                list_str = list_content.group(1)
                                # Parse list contents (handle different types)
                                if not list_str or list_str == "null":
                                    row_data[col_name] = None
                                else:
                                    # Split and convert list items
                                    list_items: List[Union[None, int, float, str]] = []
                                    for item in list_str.split(","):
                                        item = item.strip()
                                        if not item or item == "null":
                                            list_items.append(None)
                                        elif item.isdigit():
                                            list_items.append(int(item))
                                        elif item.replace(".", "", 1).isdigit():
                                            list_items.append(float(item))
                                        else:
                                            list_items.append(item)
                                    row_data[col_name] = list_items
                            else:
                                row_data[col_name] = None
                        else:
                            row_data[col_name] = value

                    # Add to results
                    data.append(row_data)

                except Exception as e:
                    logger.error(f"Error processing row {i+2}: {e}")
                    # Continue with the next row
                    continue

        # Create DataFrame
        df = pd.DataFrame(data)
        logger.info(
            f"CSV reading complete! Read {len(df)} rows, {len(df.columns)} columns"
        )

        return df

    def _convert_to_numpy(self, df: pd.DataFrame, id_column: str) -> pd.DataFrame:
        """
        Convert all list columns in DataFrame to numpy arrays, replacing null values with 0

        Args:
            df: Input DataFrame
            id_column: ID column name (should be the first column)

        Returns:
            Converted DataFrame
        """
        logger.info("Converting lists in DataFrame to numpy arrays...")

        # Create a new DataFrame to store results
        result_df = pd.DataFrame(index=df.index)

        # Preserve ID column
        result_df[id_column] = df[id_column]

        # Process all other columns
        list_columns = [col for col in df.columns if col != id_column]
        total_cols = len(list_columns)

        for idx, col_name in enumerate(list_columns):
            logger.info(f"Processing column {idx+1}/{total_cols}: '{col_name}'")
            result_df[col_name] = self._convert_column(df, col_name)

        logger.info("List conversion to numpy arrays completed")
        return result_df

    def _convert_column(self, df: pd.DataFrame, column_name: str) -> pd.Series:
        """
        Convert a DataFrame column containing list data to numpy arrays

        Args:
            df: Input DataFrame
            column_name: Column name to convert

        Returns:
            Series containing numpy arrays
        """
        result = []

        for i, list_data in enumerate(df[column_name]):
            # Display progress for large datasets
            if i % 10000 == 0 and i > 0:
                logger.info(f"Processed {i} rows for column '{column_name}'...")

            try:
                # Clean string, remove \x00 characters
                if isinstance(list_data, str):
                    list_data = list_data.replace("\x00", "")

                    # Extract list content
                    match = self.list_pattern.search(list_data)
                    if match:
                        items_str = match.group(1)
                        # Split and process list items
                        items = items_str.split(",")

                        # Convert to numeric values, replace null with 0
                        numeric_items: List[Union[int, float]] = []
                        for item in items:
                            item = item.strip()
                            if item == "null" or not item:
                                numeric_items.append(0)
                            elif item.isdigit():
                                numeric_items.append(int(item))
                            elif item.replace(".", "", 1).isdigit():
                                numeric_items.append(float(item))
                            else:
                                numeric_items.append(
                                    0
                                )  # Replace non-numeric items with 0

                        result.append(np.array(numeric_items))
                    else:
                        result.append(np.array([]))
                elif isinstance(list_data, list):
                    # If already a list, just replace null with 0
                    numeric_items = [0 if item is None else item for item in list_data]
                    result.append(np.array(numeric_items))
                else:
                    result.append(np.array([]))

            except Exception as e:
                logger.error(f"Error processing row {i} in column '{column_name}': {e}")
                result.append(np.array([]))

        return pd.Series(result, index=df.index)
