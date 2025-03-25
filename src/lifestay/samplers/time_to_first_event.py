"""
Time-to-First Event Analysis Module

This module provides classes for analyzing time to first event occurrences
in time series data, optimized for survival analysis and user retention studies.
"""

from typing import Any, Dict, List, Optional, Tuple

from .pattern_builder import PatternBuilder


class TimeToFirstEventBuilder(PatternBuilder):
    """
    A class that builds generators to identify the first occurrence of events in time series data.

    This class creates processors that analyze time series data using sliding windows
    to detect the first occurrence of significant events for each individual. It follows
    a similar approach to HeartbeatDeclinePatternBuilder but only considers the first event.

    Event Definition:
    ----------------
    An event occurs on a given day if the relative change between two windows exceeds
    a negative threshold and doesn't meet filtering conditions (similar to HeartbeatDeclinePatternBuilder).
    Only the first occurrence of the event for each individual is considered.

    Missing data is represented with an event indicator of 0.
    """

    def __init__(
        self,
        window_size: int,
        past_window_start_offset: Optional[int] = None,
        drop_threshold: float = 0.2,
        min_avg_value: float = 0.0,
        max_avg_value: float = float("inf"),
    ):
        """
        Initialize the TimeToFirstEventBuilder with configuration parameters.

        Args:
            window_size: Number of days to include in each window
            past_window_start_offset: Starting offset for the past window relative to current day
                                     (defaults to 2 * window_size if not specified)
            drop_threshold: Threshold percentage for value decline (0.2 = 20%)
            min_avg_value: Minimum average value to consider for event triggering
            max_avg_value: Maximum average value to consider for event triggering

        Raises:
            ValueError: If input parameters are invalid
        """
        super().__init__(
            window_size=window_size,
            past_window_start_offset=past_window_start_offset,
            threshold=drop_threshold,
            min_avg_value=min_avg_value,
            max_avg_value=max_avg_value,
        )
        # For consistent naming
        self.drop_threshold = self.threshold

    def _is_event(
        self, relative_change: float, recent_window_avg: float, past_window_avg: float
    ) -> bool:
        """
        Determine if a decline event occurred based on the relative change.

        Args:
            relative_change: The calculated relative change between windows
            recent_window_avg: Average value in the recent window
            past_window_avg: Average value in the past window

        Returns:
            Boolean indicating whether a decline event occurred
        """
        return relative_change < -self.threshold

    def generate_first_event_data(
        self, individual_data: Dict[str, List[float]]
    ) -> List[Dict[str, Any]]:
        """
        Analyzes time series data for multiple individuals and identifies the first event for each.

        Args:
            individual_data: Dictionary mapping individual IDs to their time series data

        Returns:
            List of dictionaries containing time-to-first-event information for each individual:
            [
                {
                    "individual_id": str,  # ID of the individual
                    "observation_time": int,  # Total observation time (length of time series)
                    "event_time": Optional[int],  # Time index when the first event occurred (None if no event)
                    "event_indicator": int,  # 1 if event occurred, 0 if no event or missing data
                },
                ...
            ]
        """
        result = []

        for individual_id, time_series in individual_data.items():
            # Skip empty time series or handle as missing data
            if not time_series:
                result.append(
                    {
                        "individual_id": individual_id,
                        "observation_time": 0,
                        "event_time": None,
                        "event_indicator": 0,  # Missing data has event indicator 0
                    }
                )
                continue

            observation_time = len(time_series)
            event_time = None
            event_indicator = 0

            # Check if we have enough data to detect events
            if observation_time >= self.past_window_start_offset + 1:
                # Get boolean events for this individual
                try:
                    events = list(self.generate_boolean_events(time_series))

                    # Find the first True event (if any)
                    for i, is_event in enumerate(events):
                        if is_event:
                            # Convert to absolute day index (add offset)
                            event_time = i + self.past_window_start_offset
                            event_indicator = 1
                            break
                except ValueError:
                    # Not enough data to process, treat as no event
                    pass

            result.append(
                {
                    "individual_id": individual_id,
                    "observation_time": observation_time,
                    "event_time": event_time,
                    "event_indicator": event_indicator,
                }
            )

        return result

    def generate_survival_data(
        self, individual_data: Dict[str, List[float]]
    ) -> Tuple[List[int], List[int]]:
        """
        Generates survival analysis data format from the time series data.

        Args:
            individual_data: Dictionary mapping individual IDs to their time series data

        Returns:
            Tuple containing:
            - List of observation times for each individual
            - List of event indicators (1=event occurred, 0=censored/no event) for each individual
        """
        first_event_data = self.generate_first_event_data(individual_data)

        observation_times = []
        event_indicators = []

        for data in first_event_data:
            observation_times.append(data["observation_time"])
            event_indicators.append(data["event_indicator"])

        return observation_times, event_indicators
