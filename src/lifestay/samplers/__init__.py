from lifestay import logger

# Create a submodule logger
logger = logger.getChild("samplers")

# Import the EOL Extractor
from .eol_extractor import EOLExtractor
from .log_utils import setup_logger

# Import the Pattern Builder
from .pattern_builder import HeartbeatDeclinePatternBuilder

# Import the Window Averager
from .window_averager import WindowAverager

# Export public classes
__all__ = [
    "EOLExtractor",
    "setup_logger",
    "WindowAverager",
    "HeartbeatDeclinePatternBuilder",
]
