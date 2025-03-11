from lifestay import logger

# Create a submodule logger
logger = logger.getChild("samplers")

# Import the EOL Extractor
from .eol_extractor import EOLExtractor

# Import the Window Averager
from .window_averager import WindowAverager

# Export public classes
__all__ = [
    "EOLExtractor",
    "WindowAverager",
]
