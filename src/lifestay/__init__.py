import logging
import sys


# Configure logging
class lifestayFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Remove 'lifestay.' prefix from the logger name if present
        if record.name.startswith("lifestay."):
            record.name = record.name[7:]
        return super().format(record)


formatter = lifestayFormatter(
    fmt="%(asctime)s - %(name)s \t%(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",  # Format timestamp to show only seconds, not milliseconds
)

# Create console handler and set formatter
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Remove any existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
root_logger.addHandler(console_handler)

# Create package-level logger
logger = logging.getLogger("lifestay")

# Import additional functions from the survival package
from lifestay.survival import (
    basic_cox_model,
    calculate_feature_contribution,
    cox_model_with_interactions,
    generate_comprehensive_report,
    survival_risk_groups,
    visualize_contribution_distribution,
    visualize_risk_distribution,
)

# Import the CSVProcessor class
from .loader.csv_processor import CSVProcessor

# Direct imports for public functions
from .loader.load_csv import (
    convert_dataframe_lists_to_numpy,
    convert_list_column_to_numpy,
    read_csv_with_lists,
)

# Import the SampleBuilder class
from .loader.sample_builder import SampleBuilder

# Import the EOLExtractor class
from .samplers.eol_extractor import EOLExtractor

# Import the WindowAverager class
from .samplers.window_averager import WindowAverager

# Import survival analysis components
from .survival.basic_cox import basic_cox_model
from .survival.comprehensive_visualization import generate_comprehensive_report
from .survival.cox_model import CoxPHModel
from .survival.data_builder import SurvivalDatasetBuilder
from .survival.data_converter import convert_to_lifelines_format
from .survival.interaction_cox import cox_model_with_interactions
from .survival.risk_stratification import (
    survival_risk_groups,
    visualize_risk_distribution,
)
from .survival.utils import (
    create_survival_dataset_from_sample_builder,
    fit_and_evaluate_cox_model,
    plot_survival_curves,
)

# List of public objects that can be imported with "from lifestay import *"
__all__ = [
    "basic_cox_model",
    "calculate_feature_contribution",
    "convert_dataframe_lists_to_numpy",
    "convert_list_column_to_numpy",
    "convert_to_lifelines_format",
    "cox_model_with_interactions",
    "CoxPHModel",
    "create_survival_dataset_from_sample_builder",
    "CSVProcessor",
    "EOLExtractor",
    "fit_and_evaluate_cox_model",
    "generate_comprehensive_report",
    "plot_survival_curves",
    "read_csv_with_lists",
    "SampleBuilder",
    "survival_risk_groups",
    "SurvivalDatasetBuilder",
    "visualize_contribution_distribution",
    "visualize_risk_distribution",
    "WindowAverager",
]
