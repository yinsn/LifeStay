import os
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .log_utils import setup_logger

# Create a logger for this module
logger = setup_logger("lifestay.loader.load_csv")


def read_csv_with_lists(
    file_path: str, sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Read CSV file containing lists line by line

    Args:
        file_path: CSV file path
        sample_size: If specified, only read n rows (for testing)

    Returns:
        pandas DataFrame
    """
    logger.info("Starting to read file")
    logger.info(
        f"File size: {os.path.getsize(file_path) / (1024 * 1024 * 1024):.2f} GB"
    )

    # List to store all data
    data: List[Dict[str, Any]] = []

    # Pattern matching for list fields
    list_pattern = re.compile(r"\[(.*?)\]")

    # Open file and read
    with open(file_path, "r", encoding="utf-8") as f:
        # Read and process header row
        original_header = f.readline().strip().split(",")

        # Extract the part after the dot as the new header
        header = [col.split(".")[-1] if "." in col else col for col in original_header]

        id_col = header[0]  # First column is ID
        list_cols = header[1:]  # Other columns are lists

        # Save the mapping between original and new headers
        header_mapping = dict(zip(original_header, header))

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
                        list_content = list_pattern.search(value)
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
    logger.info(f"Complete! Read {len(df)} rows, {len(df.columns)} columns")

    # Print header mapping
    logger.info("\nHeader mapping relationships:")
    for orig, new in header_mapping.items():
        if orig != new:  # Only show mappings that changed
            logger.info(f"  {orig} -> {new}")

    return df


def convert_list_column_to_numpy(df: pd.DataFrame, column_name: str) -> pd.Series:
    """
    Convert a DataFrame column containing list strings to numpy arrays, replacing null values with 0

    Args:
        df: Input DataFrame
        column_name: Column name to convert

    Returns:
        Series containing numpy arrays
    """
    result = []

    # Pattern matching for list fields
    list_pattern = re.compile(r"\[(.*?)\]")

    for i, list_str in enumerate(df[column_name]):
        # Display progress
        if i % 10000 == 0 and i > 0:
            logger.info(f"Processed {i} rows...")

        try:
            # Clean string, remove \x00 characters
            if isinstance(list_str, str):
                list_str = list_str.replace("\x00", "")

                # Extract list content
                match = list_pattern.search(list_str)
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
                            numeric_items.append(0)  # Replace non-numeric items with 0

                    result.append(np.array(numeric_items))
                else:
                    result.append(np.array([]))
            elif isinstance(list_str, list):
                # If already a list, just replace null with 0
                numeric_items = [0 if item is None else item for item in list_str]
                result.append(np.array(numeric_items))
            else:
                result.append(np.array([]))

        except Exception as e:
            logger.error(f"Error processing row {i}: {e}")
            result.append(np.array([]))

    return pd.Series(result, index=df.index)


def convert_dataframe_lists_to_numpy(
    df: pd.DataFrame, id_column: str = "user_id"
) -> pd.DataFrame:
    """
    Convert all list columns in DataFrame to numpy arrays, replacing null values with 0

    Args:
        df: Input DataFrame
        id_column: ID column name, default is 'user_id'

    Returns:
        Converted DataFrame
    """
    logger.info(f"Starting to convert lists in DataFrame to numpy arrays...")
    logger.info(f"DataFrame has {len(df)} rows, {len(df.columns)} columns")

    # Create a new DataFrame to store results
    result_df = pd.DataFrame(index=df.index)

    # Preserve ID column
    result_df[id_column] = df[id_column]

    # Process all other columns
    list_columns = [col for col in df.columns if col != id_column]
    total_cols = len(list_columns)

    for idx, col_name in enumerate(list_columns):
        logger.info(f"Processing column {idx+1}/{total_cols}: '{col_name}'")
        result_df[col_name] = convert_list_column_to_numpy(df, col_name)

    logger.info(f"Complete! All columns have been converted to numpy arrays")
    return result_df
