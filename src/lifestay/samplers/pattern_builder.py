"""
Pattern Builder Module

This module provides classes for building pattern recognition generators/cursors
to identify events in time series data, specifically optimized for efficient processing
of large datasets.
"""

from typing import Any, Dict, Generator, List, Optional


class HeartbeatDeclinePatternBuilder:
    """
    A class that builds generators to identify heartbeat decline events in time series data.

    This class creates cursors (generators) that process heartbeat time series
    using sliding windows to detect significant declines in user heartbeat activity.
    It compares a recent window of heartbeat data with a past window and identifies
    events where the relative change exceeds a specified threshold.

    Event Definition:
    ----------------
    An event occurs on a given day if the relative change between two windows exceeds
    a negative threshold and doesn't meet filtering conditions:

    - Recent Window: Average heartbeat over the most recent window_size days before the current day
    - Past Window: Average heartbeat over window_size days starting from past_window_start_offset days before current day
    - Relative Change = (Recent Window Avg - Past Window Avg) / Past Window Avg
    - Event Occurs if: Relative Change < -drop_threshold AND filtering conditions are not met

    The Past Window and Recent Window will never overlap as past_window_start_offset must be at least 2 * window_size.
    """

    def __init__(
        self,
        window_size: int,
        past_window_start_offset: Optional[int] = None,
        drop_threshold: float = 0.2,
        min_avg_heartbeat: float = 0.0,
        max_avg_heartbeat: float = float("inf"),
    ):
        """
        Initialize the HeartbeatDeclinePatternBuilder with configuration parameters.

        Args:
            window_size: Number of days to include in each window
            past_window_start_offset: Starting offset for the past window relative to current day
                                     (defaults to 2 * window_size if not specified)
            drop_threshold: Threshold percentage for heartbeat decline (0.2 = 20%)
            min_avg_heartbeat: Minimum average heartbeat to consider for event triggering
            max_avg_heartbeat: Maximum average heartbeat to consider for event triggering

        Raises:
            ValueError: If input parameters are invalid
        """
        # Validate input parameters
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer")

        # Set default past_window_start_offset to 2 * window_size if not specified
        if past_window_start_offset is None:
            past_window_start_offset = 2 * window_size

        if past_window_start_offset <= 0:
            raise ValueError("past_window_start_offset must be a positive integer")

        # Ensure the windows don't overlap by requiring past_window_start_offset >= 2 * window_size
        if past_window_start_offset < 2 * window_size:
            raise ValueError(
                f"past_window_start_offset must be >= 2 * window_size to prevent window overlap. "
                f"Got past_window_start_offset={past_window_start_offset}, window_size={window_size}"
            )

        if drop_threshold <= 0:
            raise ValueError("drop_threshold must be a positive float")

        if min_avg_heartbeat < 0:
            raise ValueError("min_avg_heartbeat must be a non-negative float")

        if max_avg_heartbeat < 0:
            raise ValueError("max_avg_heartbeat must be a non-negative float")

        if min_avg_heartbeat > max_avg_heartbeat:
            raise ValueError(
                "min_avg_heartbeat cannot be greater than max_avg_heartbeat"
            )

        # Store configuration parameters
        self.window_size = window_size
        self.past_window_start_offset = past_window_start_offset
        self.drop_threshold = drop_threshold
        self.min_avg_heartbeat = min_avg_heartbeat
        self.max_avg_heartbeat = max_avg_heartbeat

    def generate_detailed_events(
        self, heartbeat_series: List[float]
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Builds a generator that identifies heartbeat decline events in a time series.

        Args:
            heartbeat_series: List of heartbeat values ordered by date (index 0 = earliest)

        Returns:
            Generator yielding dictionaries containing event information for each valid day:
            {
                "day_index": int,  # Index of the day in the original series
                "recent_window_avg": float,  # Average heartbeat in recent window
                "past_window_avg": float,  # Average heartbeat in past window
                "relative_change": float,  # Relative change between windows
                "is_event": bool  # Whether a decline event occurred
            }

        Raises:
            ValueError: If there is not enough data to process

        Example:
            >>> builder = HeartbeatDeclinePatternBuilder(window_size=2, past_window_start_offset=4, drop_threshold=0.2)
            >>> heartbeat_data = [10, 12, 15, 14, 13, 10, 8, 5, 4, 3]
            >>> pattern_events = builder.generate_detailed_events(heartbeat_data)
            >>> for event in pattern_events:
            ...     print(f"Day {event['day_index']}: Event = {event['is_event']}")
        """
        # Check if we have enough data
        data_length = len(heartbeat_series)
        min_required_data = self.past_window_start_offset + 1

        if data_length < min_required_data:
            raise ValueError(
                f"Not enough data points. Need at least {min_required_data} points, "
                f"but got {data_length}"
            )

        # Process each day starting from the earliest day where we can calculate both windows
        for current_day in range(self.past_window_start_offset, data_length):
            # Calculate indices for recent window - include current day
            recent_window_end = current_day + 1  # exclusive
            recent_window_start = current_day - self.window_size + 1  # inclusive

            # Calculate indices for past window
            # When past_window_start_offset = 2*window_size, the past window should be adjacent to the recent window
            past_window_end = recent_window_start  # exclusive
            past_window_start = past_window_end - self.window_size  # inclusive

            # Check if we have valid windows
            if past_window_start < 0 or recent_window_start < 0:
                continue

            # Calculate average heartbeats for both windows
            recent_window_avg = (
                sum(heartbeat_series[recent_window_start:recent_window_end])
                / self.window_size
            )
            past_window_avg = (
                sum(heartbeat_series[past_window_start:past_window_end])
                / self.window_size
            )

            # Handle division by zero
            if past_window_avg == 0:
                if recent_window_avg == 0:
                    # Both averages are 0, no relative change
                    relative_change = 0.0
                else:
                    # Recent window has activity but past window doesn't
                    # This is an increase, not a decrease, so we won't trigger an event
                    relative_change = float("inf")
            else:
                # Calculate relative change
                relative_change = (
                    recent_window_avg - past_window_avg
                ) / past_window_avg

            # Determine if event occurred
            # Check relative change threshold
            meets_threshold = relative_change < -self.drop_threshold

            # Check filter conditions
            filtered_by_min = (
                recent_window_avg <= self.min_avg_heartbeat
                and past_window_avg <= self.min_avg_heartbeat
            )
            filtered_by_max = (
                recent_window_avg >= self.max_avg_heartbeat
                and past_window_avg >= self.max_avg_heartbeat
            )

            # Event occurs if threshold is met and filter conditions are not
            is_event = meets_threshold and not filtered_by_min and not filtered_by_max

            # Yield the result for this day
            yield {
                "day_index": current_day,
                "recent_window_avg": recent_window_avg,
                "past_window_avg": past_window_avg,
                "relative_change": relative_change,
                "is_event": is_event,
            }

    def generate_boolean_events(
        self, heartbeat_series: List[float]
    ) -> Generator[bool, None, None]:
        """
        A simplified generator that only yields boolean values indicating whether
        a decline event occurred on each day.

        Args:
            heartbeat_series: List of heartbeat values ordered by date (index 0 = earliest)

        Returns:
            Generator yielding boolean values, True if a decline event occurred on that day

        Raises:
            ValueError: If there is not enough data to process

        Example:
            >>> builder = HeartbeatDeclinePatternBuilder(window_size=2, past_window_start_offset=4, drop_threshold=0.2)
            >>> heartbeat_data = [10, 12, 15, 14, 13, 10, 8, 5, 4, 3]
            >>> pattern_events = builder.generate_boolean_events(heartbeat_data)
            >>> events = list(pattern_events)
            >>> print(events)  # [False, True, True, False, ...]
        """
        # Leverage the detailed events generator and extract just the is_event flag
        pattern_events = self.generate_detailed_events(heartbeat_series)

        # Yield only the boolean event indicator
        for event_data in pattern_events:
            yield event_data["is_event"]

    def generate_boolean_events_list(self, heartbeat_series: List[float]) -> List[bool]:
        """
        Creates a list of boolean values aligned with the heartbeat_series length,
        indicating whether a decline event occurred on each day.

        Args:
            heartbeat_series: List of heartbeat values ordered by date (index 0 = earliest)

        Returns:
            List of boolean values with the same length as heartbeat_series.
            Positions where events couldn't be calculated (due to insufficient history)
            are set to False.

        Raises:
            ValueError: If there is not enough data to process

        Example:
            >>> builder = HeartbeatDeclinePatternBuilder(window_size=2, drop_threshold=0.2)
            >>> heartbeat_data = [10, 12, 15, 14, 13, 10, 8, 5, 4, 3]
            >>> events = builder.generate_boolean_events_list(heartbeat_data)
            >>> print(events)  # [False, False, False, False, False, True, True, True, True, True]
        """
        # Initialize result list with False values
        result = [False] * len(heartbeat_series)

        # Get boolean events and fill in the appropriate positions
        boolean_events = list(self.generate_boolean_events(heartbeat_series))
        start_idx = self.past_window_start_offset

        # Fill the result list starting from the position where we have valid events
        for i, event in enumerate(boolean_events):
            result[start_idx + i] = event

        return result
