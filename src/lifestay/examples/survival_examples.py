"""
Survival Analysis Examples Module

This module provides examples of how to use the SurvivalDatasetBuilder and CoxPHModel classes.
"""

import pandas as pd

from ..samplers.eol_extractor import EOLExtractor
from ..samplers.window_averager import WindowAverager
from ..survival.cox_model import CoxPHModel
from ..survival.data_builder import SurvivalDatasetBuilder
from ..survival.utils import fit_and_evaluate_cox_model, plot_survival_curves


def example_functional_style() -> None:
    """
    Example of using SurvivalDatasetBuilder in a functional style.
    """
    # Create builder with required parameters
    builder = SurvivalDatasetBuilder(
        feature_columns=["col1", "col2", "col3"],
        heartbeat_column="heartbeat",
        patterns={
            "negative": [1, 0, 0],  # Event occurred
            "positive": [1, 1, 1],  # Censored
        },
    )

    # Use the builder as a function
    try:
        # This requires 'your_data.csv' to exist
        model = builder(file_path="your_data.csv", sample_size=10000)

        # Get X, T, E from the model
        X, T, E = model.X, model.T, model.E

        print(f"Dataset created with {len(X)} samples")
        print(f"Features: {X.columns.tolist()}")
        print(f"Events: {E.sum()} occurred, {len(E) - E.sum()} censored")

        # Fit and evaluate model
        model, train_metrics, test_metrics = fit_and_evaluate_cox_model(X, T, E)

        # Print results
        print("\nModel Summary:")
        model.print_summary()

        print("\nTraining Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.4f}")

        print("\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")

    except FileNotFoundError:
        print("Note: This example requires 'your_data.csv' to actually run")
    except Exception as e:
        print(f"Error: {str(e)}")


def example_method_chaining() -> None:
    """
    Example of using SurvivalDatasetBuilder with method chaining.
    """

    # Create a custom pre-processing hook
    def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric features to have zero mean and unit variance."""
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if col.endswith("_idx") or col == "sample_type":
                continue
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        return df

    # Create builder and chain methods
    try:
        # Configure the builder with method chaining
        builder = (
            SurvivalDatasetBuilder(
                feature_columns=["col1", "col2", "col3"],
                heartbeat_column="heartbeat",
                patterns={"negative": [1, 0, 0], "positive": [1, 1, 1]},
            )
            .with_file("your_data.csv")
            .with_sample_size(10000)
            .with_debug(True)
            .add_pre_process_hook(normalize_features)
        )

        # Build the dataset
        model = builder.build()

        # Get X, T, E from the model
        X, T, E = model.X, model.T, model.E

        print(f"Dataset created with {len(X)} samples")
        print(f"Features: {X.columns.tolist()}")
        print(f"Events: {E.sum()} occurred, {len(E) - E.sum()} censored")

        # Create and fit the model
        model = CoxPHModel().fit(X, T, E)

        # Print model summary
        model.print_summary()

    except FileNotFoundError:
        print("Note: This example requires 'your_data.csv' to actually run")
    except Exception as e:
        print(f"Error: {str(e)}")


def example_config_based() -> None:
    """
    Example of using SurvivalDatasetBuilder with configuration dictionaries.
    """
    # Define a configuration dictionary
    config = {
        "feature_columns": ["col1", "col2", "col3"],
        "heartbeat_column": "heartbeat",
        "patterns": {"negative": [1, 0, 0], "positive": [1, 1, 1]},
        "window_size": 5,
        "event_value": "negative",
        "debug": True,
        "file_path": "your_data.csv",
        "sample_size": 10000,
    }

    try:
        # Create builder from config
        builder = SurvivalDatasetBuilder.from_config(config)

        # Build the dataset
        model = builder.build()

        # Get X, T, E from the model
        X, T, E = model.X, model.T, model.E

        print(f"Dataset created with {len(X)} samples")
        print(f"Features: {X.columns.tolist()}")
        print(f"Events: {E.sum()} occurred, {len(E) - E.sum()} censored")

        # Create and fit the model
        model = CoxPHModel().fit(X, T, E)

        # Print model summary
        model.print_summary()

    except FileNotFoundError:
        print("Note: This example requires 'your_data.csv' to actually run")
    except Exception as e:
        print(f"Error: {str(e)}")


def example_direct_time_to_event() -> None:
    """
    Example demonstrating the use of direct time-to-event calculation
    using the first non-zero index and EOL index.
    """
    try:
        # This example requires 'user_data.csv' to exist
        # The file should contain a column with heartbeat arrays for each user
        # Load your data
        df = pd.read_csv("user_data.csv")

        # Assuming the heartbeat column contains NumPy arrays
        # Convert string representations to actual NumPy arrays if needed
        # df['heartbeat'] = df['heartbeat'].apply(lambda x: np.array(eval(x)))

        print("Step 1: Extract EOL patterns and first non-zero indices")
        # Initialize the EOL extractor
        eol_extractor = EOLExtractor(default_pattern=[1, 0, 0])

        # Find EOL patterns and first non-zero indices
        # This will add two columns: 'heartbeat_eol_indices' and 'heartbeat_first_nonzero_idx'
        df_with_patterns = eol_extractor.find_eol_patterns(
            df=df,
            heartbeat_column="heartbeat",
            pattern=[1, 0, 0],  # Non-zero value followed by two zeros
        )

        print("Step 2: Calculate window averages with time to event")
        # Initialize the window averager
        window_averager = WindowAverager()

        # Calculate window averages and include the time to event column
        # This uses the first non-zero index to calculate time to event
        result_df = window_averager.calculate_window_averages(
            df=df_with_patterns,
            target_columns=["feature1", "feature2", "feature3"],  # Your feature columns
            eol_column="heartbeat_eol_indices",
            window_size=5,
            first_nonzero_column="heartbeat_first_nonzero_idx",  # This enables time_to_event calculation
        )

        print("Step 3: Use the result directly in survival analysis")
        # The result_df now contains a 'time_to_event' column that can be used directly
        # in survival analysis, no need for pre-process hooks

        # Create a features DataFrame
        X = result_df[["feature1_avg", "feature2_avg", "feature3_avg"]]

        # Get the time to event
        T = result_df["time_to_event"]

        # Create a dummy event indicator (1 = event occurred, 0 = censored)
        # In a real scenario, you would determine this based on your data
        E = pd.Series(1, index=result_df.index)  # Assuming all are events for example

        # Fit a Cox model
        model = CoxPHModel()

        # Create a DataFrame for lifelines
        lifelines_df = X.copy()
        lifelines_df["duration"] = T
        lifelines_df["event"] = E

        # Fit the model
        model.model.fit(df=lifelines_df, duration_col="duration", event_col="event")

        model.X = X
        model.T = T
        model.E = E
        model.fitted = True

        # Print model summary
        print("\nModel Summary:")
        model.print_summary()

        # Plot survival curves
        plot_survival_curves(model)

    except FileNotFoundError:
        print("Example file not found. This is just an example of the API usage.")
    except ValueError as e:
        print(f"Error in example: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    print("Example 1: Functional Style")
    print("-" * 50)
    example_functional_style()

    print("\nExample 2: Method Chaining")
    print("-" * 50)
    example_method_chaining()

    print("\nExample 3: Config Based")
    print("-" * 50)
    example_config_based()

    print("\nExample 4: Direct Time-to-Event Calculation")
    print("-" * 50)
    example_direct_time_to_event()
